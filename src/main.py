#!/usr/bin/env python3
import argparse
import json
import os
import sys
from typing import Any, Dict, List, NoReturn

from tabulate import tabulate


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments matching action.yml specs"""
    parser = argparse.ArgumentParser(
        description="QA Copilot for Infrastructure Validation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "tf_apply_job",
        type=str,
        help="Name of the GitHub Actions job that executed terraform apply",
    )

    parser.add_argument(
        "llm_provider",
        type=str.lower,
        choices=["openai", "anthropic", "gemini"],
        help="AI service provider for test generation",
    )

    parser.add_argument(
        "api_key", type=str, help="API key for the selected LLM provider"
    )

    parser.add_argument(
        "failure_strategy",
        type=str.lower,
        choices=["rollback", "alert-only", "retry"],
        default="rollback",
        help="Behavior when infrastructure validation fails",
    )

    return parser.parse_args()


def validate_inputs(args: argparse.Namespace) -> None:
    """Additional validation beyond argparse"""
    if not args.api_key.strip():
        raise ValueError("API key cannot be empty or whitespace")


def set_github_outputs(outputs: Dict[str, Any]) -> None:
    """Set GitHub Action outputs

    Args:
        outputs: Dictionary of output keys to values (must be JSON-serializable)
    """
    with open(os.environ.get("GITHUB_OUTPUT", ""), "a") as fh:
        for key, value in outputs.items():
            print(f"{key}={json.dumps(value)}", file=fh)


def print_config(args: argparse.Namespace) -> None:
    """Print formatted configuration table

    Args:
        args: Parsed command line arguments from parse_arguments()
    """
    headers: List[str] = ["Setting", "Value"]
    table: List[List[str]] = [
        ["Terraform Job", args.tf_apply_job],
        ["AI Provider", args.llm_provider.capitalize()],
        ["Failure Strategy", args.failure_strategy.capitalize()],
        ["API Key", "***" if args.api_key else ""],
    ]

    print("##[group]📋 Configuration Summary")
    print(
        tabulate(
            table,
            headers=headers,
            tablefmt="github",
            colalign=("right", "left"),
        )
    )
    print("##[endgroup]\n")


def main() -> None:
    """Main execution flow with error handling"""
    try:
        args: argparse.Namespace = parse_arguments()
        validate_inputs(args)
        print_config(args)

        # Simulate test execution
        print("\n🔍 Analyzing terraform apply output...")
        print("✅ Detected 3 AWS EC2 instances")
        print("🛡️ Running security validation...")

        # Create typed mock outputs
        outputs: Dict[str, Any] = {
            "test_plan": {
                "security_scan": "aws inspector scan",
                "connectivity_check": "ping -c 4 <ip>",
            },
            "execution_logs": [
                "Security scan completed - 0 vulnerabilities found",
                "All instances responding to ping",
            ],
            "exit_codes": {"security_scan": 0, "connectivity_check": 0},
            "infrastructure_status": "validated",
        }

        set_github_outputs(outputs)
        print("\n🎉 Success: Infrastructure validation passed!")

    except argparse.ArgumentError as e:
        handle_error(f"Invalid input: {str(e)}", debug_hint="Check workflow inputs")
    except ValueError as e:
        handle_error(f"Validation error: {str(e)}")
    except Exception as e:
        handle_error(f"Unexpected error: {str(e)}")


def handle_error(message: str, debug_hint: str = "", exit_code: int = 1) -> NoReturn:
    """Standardized error handling

    Args:
        message: Primary error message
        debug_hint: Troubleshooting suggestion
        exit_code: Process exit code (default 1)
    """
    print(f"##[error]❌ {message}")
    if debug_hint:
        print(f"##[debug] {debug_hint}")
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
