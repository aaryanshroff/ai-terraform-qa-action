terraform {
  required_version = ">= 1.11.3"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

# GitHub OIDC Provider for GitHub Actions
resource "aws_iam_openid_connect_provider" "github" {
  url = "https://token.actions.githubusercontent.com"

  client_id_list = [
    "sts.amazonaws.com" # Required audience for GitHub Actions
  ]

  thumbprint_list = [
    "6938fd4d98bab03faadb97b34396831e3780aea1" # GitHub's OIDC thumbprint
  ]
}

# IAM Role for GitHub Actions CI/CD
resource "aws_iam_role" "github_actions" {
  name               = "GitHubActionsHeimdall"
  description        = "Role for GitHub Actions CI/CD pipelines"
  assume_role_policy = data.aws_iam_policy_document.assume_role.json
}

# Trust policy for GitHub Actions
data "aws_iam_policy_document" "assume_role" {
  statement {
    effect  = "Allow"
    actions = ["sts:AssumeRoleWithWebIdentity"]

    principals {
      type        = "Federated"
      identifiers = [aws_iam_openid_connect_provider.github.arn]
    }

    condition {
      test     = "StringEquals"
      variable = "token.actions.githubusercontent.com:aud"
      values   = ["sts.amazonaws.com"]
    }

    condition {
      test     = "StringLike"
      variable = "token.actions.githubusercontent.com:sub"
      values = [
        "repo:aaryanshroff/heimdall-ai:*" # Allow all workflows in this repo
      ]
    }
  }
}

# Base permissions for test workflows
resource "aws_iam_role_policy" "ci_permissions" {
  name   = "HeimdallCIBasePermissions"
  role   = aws_iam_role.github_actions.id
  policy = data.aws_iam_policy_document.ci_permissions.json
}

data "aws_iam_policy_document" "ci_permissions" {
  statement {
    effect = "Allow"
    actions = [
      "s3:*",
      "dynamodb:*",
      "lambda:*",
      "sts:GetCallerIdentity"
    ]
    resources = ["*"]
  }
}
