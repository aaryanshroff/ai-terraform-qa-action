import argparse
import json
import os
import subprocess
import sys
from typing import Any, Dict, List, NoReturn, Tuple

import requests
from github import Github
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
    parser.add_argument("tf_apply_job_id", help="Name of the Terraform apply job")
    parser.add_argument("llm_provider", choices=["openai", "anthropic", "gemini"])
    parser.add_argument("api_key", help="LLM provider API key")
    parser.add_argument("failure_strategy", choices=["rollback", "alert-only", "retry"])
    return parser.parse_args()


def get_job_logs(job_name: str) -> str:
    """Retrieve raw logs from GitHub job"""
    gh = Github(os.getenv("GITHUB_TOKEN"))
    repo = gh.get_repo(os.getenv("GITHUB_REPOSITORY"))
    run_id = int(os.getenv("GITHUB_RUN_ID"))

    workflow_run = repo.get_workflow_run(run_id)
    for job in workflow_run.jobs():
        print("Job:", job.name)
        if job.name == job_name:
            return job.logs().decode("utf-8")
    raise ValueError(f"Job {job_name} not found")


def get_llm(llm_provider: str, api_key: str) -> Any:
    """Initialize LLM with structured output support"""
    if llm_provider == "gemini":
        return ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp", temperature=0, api_key=api_key
        )
    elif llm_provider == "openai":
        return ChatOpenAI(model="gpt-4-turbo", temperature=0, api_key=api_key).bind(
            response_format={"type": "json_object"}
        )
    raise ValueError(f"Unsupported provider: {llm_provider}")


def generate_validation_commands(llm: Any, logs: str) -> List[str]:
    """Generate validation commands from raw logs"""
    prompt = ChatPromptTemplate.from_template(
        """
    Analyze these Terraform execution logs and generate AWS CLI validation commands:
    
    {logs}
    
    Focus on:
    1. Resources created/modified
    2. Security configurations
    3. Network accessibility
    4. Compliance checks
    
    Return only executable bash commands in JSON format.
    """
    )

    try:
        structured_llm = llm.with_structured_output(ValidationCommands)
        chain = prompt | structured_llm
        response = chain.invoke({"logs": logs})
        return response.commands
    except Exception as e:
        raise RuntimeError(f"Command generation failed: {str(e)}")


def execute_validation(commands: List[str]) -> List[Dict[str, Any]]:
    """Execute generated commands safely with verbose logging"""
    results = []
    print(f"##[group]🔍 Executing {len(commands)} validation commands")

    for idx, cmd in enumerate(commands, 1):
        try:
            # Print command with number and formatting
            print(f"##[command]🚀 Command {idx}/{len(commands)}: {cmd}")

            result = subprocess.run(
                cmd, shell=True, check=True, timeout=30, capture_output=True, text=True
            )

            # Print success with output
            print(f"✅ Command {idx} passed")
            print(f"##[details] Output:\n{result.stdout.strip()}")

            results.append(
                {"command": cmd, "passed": True, "output": result.stdout.strip()}
            )

        except subprocess.CalledProcessError as e:
            # Print failure details with error output
            print(f"##[error]❌ Command {idx} failed (exit code {e.returncode})")
            print(f"##[details] Error output:\n{e.stderr.strip()}")

            results.append(
                {"command": cmd, "passed": False, "output": e.stderr.strip()}
            )

        except subprocess.TimeoutExpired:
            print(f"##[error]⌛ Command {idx} timed out after 30 seconds")
            results.append(
                {"command": cmd, "passed": False, "output": "Command timed out"}
            )

        print("-" * 80)  # Separator between commands

    print("##[endgroup]")
    return results


def handle_failure(strategy: str) -> NoReturn:
    """Handle failure based on selected strategy"""
    if strategy == "rollback":
        print("🚨 Critical failure - initiating rollback")
        sys.exit(1)
    elif strategy == "retry":
        print("🔄 Validation failed - restarting...")
        sys.exit(2)
    else:
        print("⚠️ Validation failed - alerting team")
        sys.exit(0)


def log_arguments(args: argparse.Namespace) -> None:
    """Log input arguments with sensitive data masking"""
    print("##[group]⚙️ Input Arguments")

    # Define sensitive fields to mask
    SENSITIVE_FIELDS = {"api_key"}

    for arg_name, arg_value in vars(args).items():
        # Mask sensitive values
        if arg_name in SENSITIVE_FIELDS:
            masked_value = "***" + str(arg_value)[-4:] if arg_value else "******"
            print(f"{arg_name.replace('_', ' ').title():<20}: {masked_value}")
        else:
            print(f"{arg_name.replace('_', ' ').title():<20}: {arg_value}")

    print("##[endgroup]")


def set_github_outputs(outputs: Dict[str, Any]) -> None:
    """Write outputs to GITHUB_OUTPUT"""
    with open(os.environ.get("GITHUB_OUTPUT", ""), "a") as f:
        for key, value in outputs.items():
            f.write(f"{key}={json.dumps(value)}\n")


def main() -> None:
    """Main execution flow"""
    try:
        args = parse_arguments()
        log_arguments(args)

        llm = get_llm(args.llm_provider, args.api_key)

        # Get logs from Terraform apply job
        print("##[group]📥 Retrieving Terraform Logs")
        raw_logs = get_job_logs(args.tf_apply_job_id)
        print(f"Retrieved {len(raw_logs.splitlines())} lines of logs")
        print("##[endgroup]")

        # Generate commands
        print("##[group]🤖 AI-Generated Validation Commands")
        commands = generate_validation_commands(
            llm, raw_logs[:100000]
        )  # Truncate to 100k chars
        print("\n".join([f"{i+1}. {cmd}" for i, cmd in enumerate(commands)]))
        print("##[endgroup]")

        # Execute validation
        results = execute_validation(commands)

        # Print summary table
        print("\n##[group]📊 Validation Summary")
        table = []
        for i, result in enumerate(results, 1):
            status = "PASS" if result["passed"] else "FAIL"
            table.append(
                [
                    i,
                    status,
                    result["command"],
                    (
                        result["output"][:100] + "..."
                        if len(result["output"]) > 100
                        else result["output"]
                    ),
                ]
            )

        print(
            tabulate(
                table,
                headers=["#", "Status", "Command", "Output Snippet"],
                tablefmt="github",
                maxcolwidths=[None, None, 50, 30],
            )
        )
        print("##[endgroup]")

        # Prepare outputs
        exit_codes = {f"check_{i}": int(not r["passed"]) for i, r in enumerate(results)}
        passed = all(r["passed"] for r in results)

        set_github_outputs(
            {
                "test_plan": [r["command"] for r in results],
                "execution_logs": [
                    f"{'PASS' if r['passed'] else 'FAIL'}: {r['command']}"
                    for r in results
                ],
                "exit_codes": exit_codes,
                "success_rate": sum(r["passed"] for r in results) / len(results),
                "infrastructure_status": "validated" if passed else "needs_fixes",
            }
        )

        if not passed:
            handle_failure(args.failure_strategy)

    except Exception as e:
        print(f"##[error]❌ Critical error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
