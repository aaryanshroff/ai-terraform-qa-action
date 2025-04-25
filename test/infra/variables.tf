variable "aws_region" {
  description = "AWS region for test resources"
  type        = string
  default     = "us-west-2"
}

variable "resource_prefix" {
  description = "Prefix for test resource names"
  type        = string
  default     = "heimdall-test"
}

variable "test_case" {
  description = "Identifier for test scenario"
  type        = string
  default     = "default"
}
