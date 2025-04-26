terraform {
  required_version = ">= 1.11.3"
  
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.6"
    }
  }

  backend "local" {
    path = "terraform.tfstate"
  }
}

provider "aws" {
  region = var.aws_region
}

resource "random_pet" "suffix" {
  length = 2
}

resource "aws_s3_bucket" "validation_test" {
  bucket = "${var.resource_prefix}-${random_pet.suffix.id}"

  tags = {
    Environment = "ValidationTest"
    ManagedBy   = "HeimdallAI"
  }
}

resource "aws_s3_bucket_ownership_controls" "validation_test" {
  bucket = aws_s3_bucket.validation_test.id

  rule {
    object_ownership = "BucketOwnerEnforced"
  }
}

resource "aws_s3_bucket_versioning" "validation_test" {
  bucket = aws_s3_bucket.validation_test.id

  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_dynamodb_table" "validation_test" {
  name           = "${var.resource_prefix}-${random_pet.suffix.id}"
  billing_mode   = "PAY_PER_REQUEST"
  hash_key       = "TestKey"

  attribute {
    name = "TestKey"
    type = "S"
  }

  server_side_encryption {
    enabled = true
  }

  tags = {
    Environment = "ValidationTest"
    ManagedBy   = "HeimdallAI"
  }
}