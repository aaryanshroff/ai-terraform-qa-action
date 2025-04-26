import argparse
import json
import os
import subprocess
import sys
from typing import Any, Dict, List, NoReturn, Tuple

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
    parser.add_argument("tf_apply_job", help="Name of the Terraform apply job")
    parser.add_argument("llm_provider", choices=["openai", "anthropic", "gemini"])
    parser.add_argument("api_key", help="LLM provider API key")
    parser.add_argument("failure_strategy", choices=["rollback", "alert-only", "retry"])
    return parser.parse_args()


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


def generate_validation_commands(llm: Any, tf_output: Dict[str, Any]) -> List[str]:
    """Generate validation commands using structured LLM output"""
    prompt = ChatPromptTemplate.from_template(
        """
    Generate AWS CLI/jq commands to validate these resources:
    {resources}
    
    Focus on:
    - Security configurations
    - Encryption status
    - Public access
    - Compliance with best practices
    - Cost optimization opportunities
    
    Return only executable bash commands in valid JSON format.
    """
    )

    try:
        structured_llm = llm.with_structured_output(ValidationCommands)
        chain = prompt | structured_llm
        response = chain.invoke({"resources": json.dumps(tf_output)})
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


def set_github_outputs(outputs: Dict[str, Any]) -> None:
    """Write outputs to GITHUB_OUTPUT"""
    with open(os.environ.get("GITHUB_OUTPUT", ""), "a") as f:
        for key, value in outputs.items():
            f.write(f"{key}={json.dumps(value)}\n")


def main() -> None:
    """Main execution flow"""
    try:
        args = parse_arguments()
        llm = get_llm(args.llm_provider, args.api_key)

        # Get Terraform outputs from environment
        tf_output = json.loads(os.environ.get("TF_OUTPUT", "{}"))
        print("##[group]📄 Terraform Outputs")
        print(json.dumps(tf_output, indent=2))
        print("##[endgroup]")

        # Generate commands
        print("##[group]🤖 AI-Generated Validation Commands")
        commands = generate_validation_commands(llm, tf_output)
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
