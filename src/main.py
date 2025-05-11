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
    This function fetches the logs by using the job's logs_url,
    which redirects to a ZIP archive of the job's logs.
    It then extracts the textual content from the files within that archive.
    """
    github_token = os.getenv("GITHUB_TOKEN")
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
    except Exception as e:
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
        all_jobs_in_run = list(workflow_run.jobs())
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
                f"Found target job: '{target_job.name}' (ID: {target_job.id}, Status: '{target_job.status}', Logs URL: {target_job.logs_url()})"
            )
            break

    if not target_job:
        print(
            f"Job '{job_name}' not found. Available jobs in run {run_id} ({len(all_jobs_in_run)} total):"
        )
        for j_idx, j in enumerate(all_jobs_in_run):
            print(f"  {j_idx+1}. Name: '{j.name}', ID: {j.id}, Status: '{j.status}'")
            if j_idx >= 19 and len(all_jobs_in_run) > 20:
                print(f"  ... and {len(all_jobs_in_run) - (j_idx+1)} more jobs.")
                break
        raise ValueError(
            f"Job '{job_name}' not found in workflow run ID '{run_id}'. "
            "Please ensure the job name matches exactly (it is case-sensitive)."
        )

    print(
        f"Attempting to download logs for job '{target_job.name}' (ID: {target_job.id}) using logs_url: {target_job.logs_url()}"
    )
    final_log_content: str = ""
    try:
        print(f"Requesting log archive from: {target_job.logs_url()}")

        log_download_response = requests.get(
            target_job.logs_url(),
            timeout=(
                10,
                180,
            ),  # (connect_timeout, read_timeout) - 10s to connect, 180s to download
        )

        # This will raise an HTTPError if the final download fails
        log_download_response.raise_for_status()

        final_log_content = log_download_response.content
        print(
            f"Successfully downloaded log archive (Final URL: {log_download_response.url}, Status: {log_download_response.status_code}, Size: {len(final_log_content)} chars)."
        )

    except requests.exceptions.Timeout as e:
        url_timed_out = e.request.url
        print(f"Timeout occurred while requesting: {url_timed_out}")
        raise RuntimeError(
            f"Timeout while trying to download log zip (URL: {url_timed_out}): {e}"
        )
    except requests.exceptions.HTTPError as e:
        url_errored = e.request.url
        print(f"HTTP error {e.response.status_code} for URL: {url_errored}")
        print(f"Response content (first 500 chars): {e.response.text[:500]}...")
        # Check if the error was from the original GitHub URL or the redirected one
        if target_job.logs_url() in url_errored:
            print("The error seems to be from the initial GitHub API logs_url.")
        else:
            print("The error seems to be from the redirected storage URL.")
        raise RuntimeError(
            f"Failed to download log zip due to HTTP error {e.response.status_code} from {url_errored}: {e}"
        )
    except requests.exceptions.RequestException as e:
        url_errored = e.request.url if e.request else "Unknown URL"
        print(
            f"Network or request error when trying to download log zip (URL: {url_errored}): {type(e).__name__} - {e}"
        )
        raise RuntimeError(f"Failed to download log zip due to a request error: {e}")
    except Exception as e:
        print(
            f"An unexpected error occurred in log download/processing for job ID {target_job.id}: {type(e).__name__} {e}"
        )
        raise RuntimeError(
            f"An unexpected error occurred while trying to get/process log zip for job ID {target_job.id}: {type(e).__name__} {e}"
        )

    return final_log_content


def get_llm(llm_provider: str, api_key: str) -> Any:
    """Initialize LLM with structured output support"""
    if llm_provider == "gemini":
        return ChatGoogleGenerativeAI(
            model="gemini-1.5-flash-latest",
            temperature=0,
            google_api_key=api_key,  # Corrected param name
        )
    elif llm_provider == "openai":
        return ChatOpenAI(model="gpt-4-turbo", temperature=0, api_key=api_key).bind(
            response_format={"type": "json_object"}
        )
    elif llm_provider == "anthropic":
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
        if not hasattr(response, "commands") or not isinstance(response.commands, list):
            print(
                f"Warning: LLM response format is incorrect or 'commands' is not a list. Response: {response}"
            )
            return []
        if not response.commands:
            print("Warning: LLM generated an empty list of commands.")
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
                text=True,
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
                timeout_output += f"\nPartial STDOUT:\n{e.stdout.decode(errors='replace').strip()}"  # stdout is bytes
            if e.stderr:
                timeout_output += f"\nPartial STDERR:\n{e.stderr.decode(errors='replace').strip()}"  # stderr is bytes
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
        with open(github_output_file, "a") as f:  # Open in append mode
            for key, value in outputs.items():
                # Sanitize key for GitHub Actions
                sanitized_key = key.replace(" ", "_").replace("-", "_")

                if isinstance(value, (dict, list)):
                    json_value = json.dumps(value)
                    # Use heredoc format for multiline JSON
                    delimiter = f"EOF_{sanitized_key.upper()}"
                    print(f"Setting output: {sanitized_key} (JSON content via heredoc)")
                    f.write(
                        f"{sanitized_key}<<{delimiter}\n{json_value}\n{delimiter}\n"
                    )
                elif isinstance(value, str) and (
                    "\n" in value
                    or "\r" in value
                    or "%" in value
                    or "'" in value
                    or '"' in value
                ):
                    # Use heredoc for any string that might be problematic
                    delimiter = f"EOF_{sanitized_key.upper()}"
                    print(
                        f"Setting output: {sanitized_key} (Multi-line/special char string via heredoc)"
                    )
                    f.write(f"{sanitized_key}<<{delimiter}\n{value}\n{delimiter}\n")
                else:
                    print(f"Setting output: {sanitized_key} = {value}")
                    f.write(f"{sanitized_key}={value}\n")
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
        if (
            not results and commands
        ):  # Commands were generated but execution yielded no results (e.g. all errored before appending)
            print(
                "No validation results to display, though commands were generated. Assuming all failed if execution block had issues."
            )
            num_failed_commands = len(commands)
        elif not commands:  # No commands were generated in the first place
            print("No validation commands were generated or executed.")
        else:  # Results exist
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

        passed_all = (
            num_failed_commands == 0 if (results or commands) else True
        )  # True if no commands were ever intended/generated
        success_rate = 0.0
        if results:  # Calculate success rate only if there were results
            success_rate = (
                (len(results) - num_failed_commands) / len(results)
                if len(results) > 0
                else 1.0
            )
        elif not commands:  # No commands to run means 100% success of doing nothing.
            success_rate = 1.0

        github_outputs = {
            "validation_commands_generated": commands,  # List of strings
            "validation_results_summary": [  # List of objects
                {
                    "command": r["command"],
                    "status": "passed" if r["passed"] else "failed",
                    "output_snippet": (
                        r.get("output", "")[:200] + "..."
                        if len(r.get("output", "")) > 200
                        else r.get("output", "")
                    ),
                }
                for r in results
            ],
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

        if not passed_all and (
            results or commands
        ):  # Only trigger failure strategy if commands were attempted
            handle_failure(args.failure_strategy, num_failed_commands)

        print("✅ AI-Powered Infrastructure Validation script completed.")
        if not passed_all and not (results or commands):
            print(
                "Note: 'passed_all' is false but no commands were run/generated; this might indicate an earlier setup issue not caught by the failure strategy or empty logs leading to no commands."
            )

    except ValueError as ve:  # Config/setup errors
        print(f"##[error]Configuration Error: {str(ve)}")
        set_github_outputs(
            {
                "overall_status": "error",
                "error_message": str(ve),
                "infrastructure_status": "unknown",
            }
        )
        sys.exit(10)
    except RuntimeError as rte:  # Operational errors within the script's logic
        print(f"##[error]Runtime Error: {str(rte)}")
        set_github_outputs(
            {
                "overall_status": "error",
                "error_message": str(rte),
                "infrastructure_status": "unknown",
            }
        )
        sys.exit(20)
    except Exception as e:  # Any other unhandled critical error
        import traceback

        print(f"##[error]❌ Critical Unhandled Error: {type(e).__name__} - {str(e)}")
        print(f"Traceback:\n{traceback.format_exc()}")
        set_github_outputs(
            {
                "overall_status": "error",
                "error_message": f"{type(e).__name__}: {str(e)}",
                "infrastructure_status": "unknown",
            }
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
