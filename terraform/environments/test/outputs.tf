output "test_case" {
  description = "Identifier for executed test scenario"
  value       = var.test_case
}

output "s3_bucket_name" {
  description = "Name of created S3 bucket"
  value       = aws_s3_bucket.validation_test.bucket
}

output "dynamodb_table_arn" {
  description = "ARN of DynamoDB table"
  value       = aws_dynamodb_table.validation_test.arn
}

output "resource_tags" {
  description = "Tags applied to test resources"
  value = {
    s3_bucket = aws_s3_bucket.validation_test.tags
    dynamodb  = aws_dynamodb_table.validation_test.tags
  }
}
