import argparse
import json
import os
import subprocess
import sys
from typing import Any, Dict, List, NoReturn, Tuple

import requests
from github import (
    Github,
    UnknownObjectException,
    GithubException,
)
import zipfile
import io

from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from tabulate import tabulate


# Structured output schema for command generation
class ValidationCommands(BaseModel):
    commands: List[str] = Field(
        description="List of AWS CLI/jq commands for infrastructure validation",
        examples=[["aws s3api get-bucket-encryption --bucket $BUCKET"]],
    )


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="AI-Powered Infrastructure Validation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("tf_apply_job_name", help="Name of the Terraform apply job")
    parser.add_argument(
        "llm_provider",
        choices=["openai", "anthropic", "gemini"],
        help="LLM provider to use",
    )
    parser.add_argument("api_key", help="API key for the selected LLM provider")
    parser.add_argument(
        "failure_strategy",
        choices=["rollback", "alert-only", "retry"],
        help="Action to take on validation failure",
    )
    return parser.parse_args()


def get_job_logs(job_name: str) -> str:
    """
    Retrieve raw logs from a specific GitHub Actions job.
    This function fetches the logs by downloading a ZIP archive of the job's logs
    and then extracting the textual content from the files within that archive.
    """
    github_token = os.getenv("GITHUB_TOKEN")
    # Fallback for specific GitHub Actions contexts if GITHUB_TOKEN is not directly available/sufficient
    if (
        not github_token
        and os.getenv("GITHUB_ACTIONS") == "true"
        and os.getenv("ACTIONS_RUNTIME_TOKEN")
    ):
        print("Using ACTIONS_RUNTIME_TOKEN as GITHUB_TOKEN was not set.")
        github_token = os.getenv("ACTIONS_RUNTIME_TOKEN")

    if not github_token:
        raise ValueError(
            "GITHUB_TOKEN environment variable not set, and ACTIONS_RUNTIME_TOKEN is not available. "
            "Ensure the token has 'actions:read' permissions for the repository."
        )

    github_repository = os.getenv("GITHUB_REPOSITORY")
    if not github_repository:
        raise ValueError(
            "GITHUB_REPOSITORY environment variable not set (e.g., 'owner/repo')."
        )

    github_run_id_str = os.getenv("GITHUB_RUN_ID")
    if not github_run_id_str:
        raise ValueError("GITHUB_RUN_ID environment variable not set.")
    try:
        run_id = int(github_run_id_str)
    except ValueError:
        raise ValueError(f"GITHUB_RUN_ID '{github_run_id_str}' is not a valid integer.")

    token_display = (
        f"{github_token[:4]}..." if len(github_token) >= 4 else "Token too short"
    )
    print(f"Initializing GitHub client... (Token starts with '{token_display}')")
    gh = Github(github_token)

    try:
        print(f"Attempting to access repository: {github_repository}")
        repo = gh.get_repo(github_repository)
        print(f"Successfully accessed repository: {repo.full_name}")
    except UnknownObjectException:
        raise RuntimeError(
            f"Repository '{github_repository}' not found or access denied. "
            "Ensure GITHUB_TOKEN has necessary permissions (e.g., 'repo' scope or 'actions:read')."
        )
    except GithubException as e:
        raise RuntimeError(
            f"GitHub API error when getting repository '{github_repository}': {e.status} {e.data}"
        )
    except Exception as e:  # Catch any other unexpected errors during repo access
        raise RuntimeError(
            f"Failed to get repository '{github_repository}': {type(e).__name__} {e}"
        )

    try:
        print(f"Attempting to access workflow run ID: {run_id}")
        workflow_run = repo.get_workflow_run(run_id)
        print(
            f"Successfully accessed workflow run: {workflow_run.id} (Name: '{workflow_run.name}', HTML: {workflow_run.html_url})"
        )
    except UnknownObjectException:
        raise RuntimeError(
            f"Workflow run ID '{run_id}' not found in repository '{github_repository}'."
        )
    except GithubException as e:
        raise RuntimeError(
            f"GitHub API error when getting workflow run ID '{run_id}': {e.status} {e.data}"
        )
    except Exception as e:
        raise RuntimeError(
            f"Failed to get workflow run ID '{run_id}': {type(e).__name__} {e}"
        )

    target_job = None
    print(
        f"Searching for job named '{job_name}' in workflow run '{workflow_run.id}'..."
    )

    all_jobs_in_run = []
    try:
        all_jobs_in_run = list(workflow_run.jobs())  # This makes an API call
    except GithubException as e:
        raise RuntimeError(
            f"GitHub API error while listing jobs for run ID '{run_id}': {e.status} {e.data}"
        )
    except Exception as e:
        raise RuntimeError(
            f"Failed to list jobs for run ID '{run_id}': {type(e).__name__} {e}"
        )

    if not all_jobs_in_run:
        print(f"Warning: No jobs found in workflow run ID '{run_id}'.")

    for job in all_jobs_in_run:
        if job.name == job_name:
            target_job = job
            print(
                f"Found target job: '{target_job.name}' (ID: {target_job.id}, Status: '{target_job.status}')"
            )
            break

    if not target_job:
        print(
            f"Job '{job_name}' not found. Available jobs in run {run_id} ({len(all_jobs_in_run)} total):"
        )
        for j_idx, j in enumerate(all_jobs_in_run):
            print(f"  {j_idx+1}. Name: '{j.name}', ID: {j.id}, Status: '{j.status}'")
            if (
                j_idx >= 19 and len(all_jobs_in_run) > 20
            ):  # Limit output for very long lists
                print(f"  ... and {len(all_jobs_in_run) - (j_idx+1)} more jobs.")
                break
        raise ValueError(
            f"Job '{job_name}' not found in workflow run ID '{run_id}'. "
            "Please ensure the job name matches exactly (it is case-sensitive)."
        )

    print(
        f"Attempting to download logs for job '{target_job.name}' (ID: {target_job.id})."
    )
    try:
        log_zip_response = (
            target_job.get_logs_zip()
        )  # This is a requests.Response object
        log_zip_response.raise_for_status()
        print(
            f"Successfully received log archive (Status: {log_zip_response.status_code}, Size: {len(log_zip_response.content)} bytes)."
        )
    except GithubException as e:
        error_message = (
            e.data.get("message", "No specific message")
            if isinstance(e.data, dict)
            else str(e.data)
        )
        print(
            f"GitHub API error when trying to get log zip for job ID {target_job.id}: Status {e.status}, Message: {error_message}"
        )
        if e.status == 403:
            print(
                "This typically indicates insufficient permissions. Ensure the GITHUB_TOKEN has 'actions:read' scope for this repository."
            )
        raise RuntimeError(
            f"Failed to get log zip due to GitHub API error: {e.status} - {error_message}"
        )
    except requests.exceptions.HTTPError as e:
        print(
            f"HTTP error when downloading log zip for job ID {target_job.id}: {e.response.status_code}"
        )
        print(f"Response content (first 500 chars): {e.response.text[:500]}...")
        raise RuntimeError(f"Failed to download log zip: HTTP {e.response.status_code}")
    except Exception as e:
        raise RuntimeError(
            f"An unexpected error occurred while trying to get log zip for job ID {target_job.id}: {type(e).__name__} {e}"
        )

    try:
        with zipfile.ZipFile(io.BytesIO(log_zip_response.content)) as zf:
            log_file_names = zf.namelist()
            if not log_file_names:
                print(
                    "Warning: The downloaded log archive is empty (contains no files)."
                )
                return ""

            print(f"Files found in log archive: {', '.join(log_file_names)}")
            log_content_parts = []
            for file_name_in_zip in sorted(log_file_names):  # Sort for consistent order
                print(f"  Reading content from archived file: {file_name_in_zip}")
                if len(log_file_names) > 1 and len(log_content_parts) > 0:
                    log_content_parts.append(
                        f"\n--- Log content from: {file_name_in_zip} ---\n"
                    )

                file_content_bytes = zf.read(file_name_in_zip)
                try:
                    log_content_parts.append(
                        file_content_bytes.decode("utf-8", errors="replace")
                    )
                except Exception as e_decode:
                    log_content_parts.append(
                        f"\n--- Error decoding {file_name_in_zip} as UTF-8: {e_decode} ---\n"
                    )
                    print(
                        f"Warning: Could not decode file '{file_name_in_zip}' as UTF-8: {e_decode}."
                    )

            final_log_content = "".join(log_content_parts)
            if not final_log_content.strip():
                print(
                    "Warning: Extracted log content is empty or contains only whitespace."
                )
            print(
                f"Successfully extracted and decoded logs. Total length: {len(final_log_content)} characters."
            )
            return final_log_content
    except zipfile.BadZipFile:
        content_preview = log_zip_response.content[:500]
        try:
            content_preview_text = content_preview.decode("utf-8", errors="replace")
        except:
            content_preview_text = str(content_preview)
        print(
            f"Error: Downloaded content for job ID {target_job.id} is not a valid ZIP archive. Preview: {content_preview_text}"
        )
        raise RuntimeError(
            "Failed to process logs: downloaded content was not a valid ZIP archive."
        )
    except Exception as e:
        raise RuntimeError(
            f"Failed to extract or decode logs from archive for job ID {target_job.id}: {type(e).__name__} {e}"
        )


def get_llm(llm_provider: str, api_key: str) -> Any:
    """Initialize LLM with structured output support"""
    if llm_provider == "gemini":
        return ChatGoogleGenerativeAI(
            model="gemini-1.5-flash-latest", temperature=0, google_api_key=api_key
        )
    elif llm_provider == "openai":
        return ChatOpenAI(model="gpt-4-turbo", temperature=0, api_key=api_key).bind(
            response_format={"type": "json_object"}
        )
    elif llm_provider == "anthropic":
        # This is a placeholder. You'll need to install and import langchain_anthropic
        # from langchain_anthropic import ChatAnthropic
        # return ChatAnthropic(model="claude-3-opus-20240229", temperature=0, api_key=api_key)
        raise ValueError(
            f"Anthropic provider selected but its implementation is currently commented out in get_llm."
        )
    raise ValueError(f"Unsupported LLM provider: {llm_provider}")


def generate_validation_commands(llm: Any, logs: str) -> List[str]:
    """Generate validation commands from raw logs"""
    prompt_template_str = """
    You are an expert in AWS infrastructure and Terraform.
    Analyze the following Terraform execution logs. Based on the resources created, modified, or managed,
    generate a list of AWS CLI commands that can be used to validate the successful deployment and configuration
    of these resources.

    Focus on generating commands for:
    1. Verifying the existence and status of key resources.
    2. Checking critical configurations (e.g., security groups, IAM policies, S3 bucket encryption, RDS settings).
    3. Validating network accessibility where applicable (e.g., NACLs, route tables, ELB health checks).
    4. Basic compliance checks if discernible from the logs (e.g., tagging, encryption standards).

    Terraform Logs:
    ```
    {logs}
    ```

    Important:
    - The commands should be executable bash commands using AWS CLI (aws ...).
    - Use environment variables (e.g., $BUCKET_NAME, $INSTANCE_ID) if resource identifiers are dynamic and
      can be set in the execution environment. If specific names/IDs are in the logs, use them directly.
    - Ensure the output is ONLY a JSON object containing a single key "commands",
      which has a list of strings as its value. Each string is one validation command.
    - Do not include any explanations or conversational text outside the JSON structure.

    Example of desired output format:
    {{"commands": ["aws s3api get-bucket-encryption --bucket my-terraform-bucket", "aws ec2 describe-instances --instance-ids i-1234567890abcdef0 --query 'Reservations[*].Instances[*].State.Name' --output text"]}}
    """
    prompt = ChatPromptTemplate.from_template(template=prompt_template_str)

    try:
        structured_llm = llm.with_structured_output(ValidationCommands)
        chain = prompt | structured_llm
        print("Invoking LLM to generate validation commands...")
        response = chain.invoke({"logs": logs})
        if (
            not hasattr(response, "commands") or not response.commands
        ):  # Check attribute and content
            print(
                "Warning: LLM generated an empty list of commands or response format is incorrect."
            )
            return []
        return response.commands
    except Exception as e:
        print(f"Error during LLM command generation: {type(e).__name__} - {str(e)}")
        raise RuntimeError(f"LLM command generation failed: {str(e)}")


def execute_validation(commands: List[str]) -> List[Dict[str, Any]]:
    """Execute generated commands safely with verbose logging"""
    results = []
    if not commands:
        print("No validation commands to execute.")
        return results

    print(f"##[group]🔍 Executing {len(commands)} validation commands")

    for idx, cmd in enumerate(commands, 1):
        print(f"\n--- Command {idx}/{len(commands)} ---")
        print(f"##[command]🚀 Executing: {cmd}")
        try:
            process = subprocess.run(
                cmd,
                shell=True,
                check=False,
                timeout=60,
                capture_output=True,
                text=True,  # check=False to handle manually
                env=os.environ.copy(),
            )
            if process.returncode == 0:
                print(f"✅ Command {idx} PASSED")
                if process.stdout.strip():
                    print(
                        f"##[details] Output:\n{process.stdout.strip()}\n##[enddetails]"
                    )
                else:
                    print("Output: (No stdout)")
                results.append(
                    {"command": cmd, "passed": True, "output": process.stdout.strip()}
                )
            else:
                print(
                    f"##[error]❌ Command {idx} FAILED (Exit Code: {process.returncode})"
                )
                output_detail = ""
                if process.stdout.strip():
                    output_detail += f"STDOUT:\n{process.stdout.strip()}\n"
                    print(
                        f"##[details] STDOUT:\n{process.stdout.strip()}\n##[enddetails]"
                    )
                if process.stderr.strip():
                    output_detail += f"STDERR:\n{process.stderr.strip()}"
                    print(
                        f"##[details] STDERR:\n{process.stderr.strip()}\n##[enddetails]"
                    )
                else:
                    if not process.stdout.strip():
                        print("STDERR: (No stderr)")
                results.append(
                    {"command": cmd, "passed": False, "output": output_detail.strip()}
                )

        except subprocess.TimeoutExpired as e:
            print(f"##[error]⌛ Command {idx} TIMED OUT after {e.timeout} seconds")
            timeout_output = f"Command timed out after {e.timeout} seconds."
            if e.stdout:
                timeout_output += (
                    f"\nPartial STDOUT:\n{e.stdout.decode(errors='replace').strip()}"
                )
            if e.stderr:
                timeout_output += (
                    f"\nPartial STDERR:\n{e.stderr.decode(errors='replace').strip()}"
                )
            results.append({"command": cmd, "passed": False, "output": timeout_output})
        except Exception as e_exec:
            print(
                f"##[error]💥 Unexpected error executing command {idx}: {type(e_exec).__name__} - {str(e_exec)}"
            )
            results.append(
                {
                    "command": cmd,
                    "passed": False,
                    "output": f"Execution error: {str(e_exec)}",
                }
            )
    print("##[endgroup]")
    return results


def handle_failure(strategy: str, num_failed: int) -> NoReturn:
    """Handle failure based on selected strategy"""
    print(f"##[error]Validation Result: {num_failed} command(s) failed.")
    if strategy == "rollback":
        print("🚨 Critical failure - Initiating rollback strategy (exit code 1).")
        sys.exit(1)
    elif strategy == "retry":
        print(
            "🔄 Validation failed - Initiating retry strategy (exit code 78 - neutral for retry)."
        )
        sys.exit(78)
    else:  # alert-only
        print(
            "⚠️ Validation failed - Alerting team (exit code 0 to not fail the workflow step)."
        )
        sys.exit(0)


def log_arguments(args: argparse.Namespace) -> None:
    """Log input arguments with sensitive data masking"""
    print("##[group]⚙️ Input Arguments")
    SENSITIVE_FIELDS = {"api_key"}
    args_dict = vars(args)
    max_key_len = max(len(k) for k in args_dict) if args_dict else 0

    for arg_name, arg_value in args_dict.items():
        display_name = arg_name.replace("_", " ").title()
        if arg_name in SENSITIVE_FIELDS and isinstance(arg_value, str) and arg_value:
            masked_value = (
                arg_value[:2] + "***" + arg_value[-2:] if len(arg_value) > 4 else "***"
            )
        else:
            masked_value = str(arg_value)
        print(f"{display_name:<{max_key_len + 1}}: {masked_value}")
    print("##[endgroup]")


def set_github_outputs(outputs: Dict[str, Any]) -> None:
    """Write outputs to GITHUB_OUTPUT file for use in subsequent GitHub Actions steps."""
    github_output_file = os.environ.get("GITHUB_OUTPUT")
    if not github_output_file:
        print(
            "Warning: GITHUB_OUTPUT environment variable not set. Outputs will not be persisted for GitHub Actions."
        )
        print("Outputs for local inspection:", json.dumps(outputs, indent=2))
        return

    print(f"##[group]📦 Setting GitHub Actions Outputs (to {github_output_file})")
    try:
        with open(github_output_file, "a") as f:
            for key, value in outputs.items():
                if isinstance(value, (dict, list)):
                    json_value = json.dumps(value)
                    delimiter = f"EOF_{key.upper().replace('-', '_')}"
                    print(f"Setting output: {key} (JSON content)")
                    f.write(f"{key}<<{delimiter}\n{json_value}\n{delimiter}\n")
                elif isinstance(value, str) and (
                    "\n" in value or "\r" in value or "%" in value
                ):  # Special chars for multiline
                    delimiter = f"EOF_{key.upper().replace('-', '_')}"
                    print(f"Setting output: {key} (Multi-line/special char string)")
                    f.write(f"{key}<<{delimiter}\n{value}\n{delimiter}\n")
                else:
                    print(f"Setting output: {key} = {value}")
                    f.write(f"{key}={value}\n")
        print("Successfully wrote outputs to GITHUB_OUTPUT.")
    except Exception as e:
        print(
            f"##[error]Failed to write to GITHUB_OUTPUT file '{github_output_file}': {e}"
        )
    print("##[endgroup]")


def main() -> None:
    """Main execution flow"""
    try:
        args = parse_arguments()
        log_arguments(args)

        print("##[group]🔑 Initializing LLM Provider")
        llm = get_llm(args.llm_provider, args.api_key)
        print(f"LLM Provider '{args.llm_provider}' initialized.")
        print("##[endgroup]")

        print("##[group]📥 Retrieving Terraform Job Logs")
        raw_logs = get_job_logs(args.tf_apply_job_name)
        if not raw_logs.strip():
            print(
                "##[warning]Retrieved logs are empty. Cannot generate validation commands."
            )
        else:
            print(
                f"Retrieved {len(raw_logs)} characters ({len(raw_logs.splitlines())} lines) of logs for job '{args.tf_apply_job_name}'."
            )
        print("##[endgroup]")

        LOG_TRUNCATION_LIMIT = 150000
        truncated_logs = raw_logs
        if len(raw_logs) > LOG_TRUNCATION_LIMIT:
            print(
                f"##[warning]Raw logs ({len(raw_logs)} chars) exceed truncation limit ({LOG_TRUNCATION_LIMIT} chars). Truncating (keeping start)."
            )
            truncated_logs = raw_logs[:LOG_TRUNCATION_LIMIT]

        commands: List[str] = []
        if not truncated_logs.strip():
            print(
                "Skipping command generation as logs are empty or became empty after truncation."
            )
        else:
            print("##[group]🤖 Generating AI Validation Commands")
            commands = generate_validation_commands(llm, truncated_logs)
            if commands:
                print("Generated Commands:")
                for i, cmd in enumerate(commands, 1):
                    print(f"  {i}. {cmd}")
            else:
                print("No validation commands were generated by the LLM.")
            print("##[endgroup]")

        results = execute_validation(commands)

        print("\n##[group]📊 Validation Summary")
        num_failed_commands = 0
        if not results and commands:
            print(
                "No validation results to display, though commands were generated (execution might have failed)."
            )
            num_failed_commands = len(
                commands
            )  # Assume all failed if no results but commands existed
        elif not commands:
            print("No validation commands were generated or executed.")
        else:
            table_data = []
            for i, result in enumerate(results, 1):
                status = "✅ PASS" if result["passed"] else "❌ FAIL"
                if not result["passed"]:
                    num_failed_commands += 1
                output_snippet = result.get("output", "")
                if len(output_snippet) > 150:
                    output_snippet = output_snippet[:150] + "..."
                table_data.append([i, status, result["command"], output_snippet])
            headers = ["#", "Status", "Command", "Output Snippet"]
            print(
                tabulate(
                    table_data,
                    headers=headers,
                    tablefmt="github",
                    maxcolwidths=[None, None, 60, 70],
                )
            )
            print(
                f"\nSummary: {len(results) - num_failed_commands}/{len(results)} commands passed."
            )
        print("##[endgroup]")

        passed_all = num_failed_commands == 0 if results or commands else True
        success_rate = 0.0
        if results:
            success_rate = (
                (len(results) - num_failed_commands) / len(results)
                if len(results) > 0
                else 1.0
            )
        elif not commands:
            success_rate = 1.0  # No commands, so 100% success of doing nothing.

        github_outputs = {
            "validation_commands_generated": commands,
            "validation_results_summary": [
                {
                    "command": r["command"],
                    "status": "passed" if r["passed"] else "failed",
                }
                for r in results
            ],  # More concise summary
            "overall_status": "success" if passed_all else "failure",
            "passed_command_count": (
                len(results) - num_failed_commands if results else 0
            ),
            "failed_command_count": num_failed_commands,
            "total_commands_executed": len(results),
            "success_rate": round(success_rate, 4),
            "infrastructure_status": "validated" if passed_all else "needs_attention",
        }
        set_github_outputs(github_outputs)

        if not passed_all and (results or commands):
            handle_failure(args.failure_strategy, num_failed_commands)

        print("✅ AI-Powered Infrastructure Validation script completed.")
        if not passed_all and not (
            results or commands
        ):  # Edge case: no commands, but logic implies failure path if not passed_all
            print(
                "Note: 'passed_all' is false but no commands were run; typically means an earlier setup issue not caught by failure strategy."
            )

    except ValueError as ve:
        print(f"##[error]Configuration Error: {str(ve)}")
        set_github_outputs(
            {
                "overall_status": "error",
                "error_message": str(ve),
                "infrastructure_status": "unknown",
            }
        )
        sys.exit(10)
    except RuntimeError as rte:
        print(f"##[error]Runtime Error: {str(rte)}")
        set_github_outputs(
            {
                "overall_status": "error",
                "error_message": str(rte),
                "infrastructure_status": "unknown",
            }
        )
        sys.exit(20)
    except Exception as e:
        import traceback

        print(f"##[error]❌ Critical Unhandled Error: {type(e).__name__} - {str(e)}")
        print(f"Traceback:\n{traceback.format_exc()}")
        set_github_outputs(
            {
                "overall_status": "error",
                "error_message": str(e),
                "infrastructure_status": "unknown",
            }
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
