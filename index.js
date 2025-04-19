const core = require('@actions/core');
const { Octokit } = require("@octokit/rest");
const { execSync } = require('child_process');
const { GoogleGenerativeAI } = require("@google/generative-ai");
const { z } = require("zod");

// Define Zod schema for QA commands
const QaCommandSchema = z.object({
  commands: z.array(
    z.object({
      name: z.string().describe("Name of the QA check"),
      aws_service: z.string().describe("AWS service being verified"),
      command: z.string().describe("AWS CLI command to execute"),
      expected_output: z.string().describe("JSONPath expression to verify output")
    })
  ).describe("List of AWS verification commands")
});

async function analyzeLogsWithAI(logs, apiKey) {
  const genAI = new GoogleGenerativeAI(apiKey);
  const model = genAI.getGenerativeModel({ model: "gemini-pro" });

  try {
    const result = await model.generateContent({
      contents: [{
        parts: [{
          text: `Analyze Terraform apply logs and generate AWS verification commands.
Focus on:
- Newly created resources (EC2, S3, IAM, etc.)
- Security group configurations
- IAM role permissions
- Resource tagging
- Service endpoint availability

Logs: ${logs.slice(-3000)}`
        }]
      }],
      generationConfig: {
        temperature: 0.1
      },
      safetySettings: [{
        category: "HARM_CATEGORY_DANGEROUS_CONTENT",
        threshold: "BLOCK_NONE"
      }],
      tools: [{
        functionDeclarations: [{
          name: "generateQACommands",
          description: "Generate AWS verification commands",
          parameters: {
            type: "OBJECT",
            properties: {
              commands: {
                type: "ARRAY",
                items: {
                  type: "OBJECT",
                  properties: {
                    name: {
                      type: "STRING",
                      description: "Name of the QA check"
                    },
                    aws_service: {
                      type: "STRING",
                      description: "AWS service being verified"
                    },
                    command: {
                      type: "STRING",
                      description: "AWS CLI command to execute"
                    },
                    expected_output: {
                      type: "STRING",
                      description: "JSONPath expression to verify output"
                    }
                  },
                  required: ["name", "aws_service", "command", "expected_output"]
                }
              }
            },
            required: ["commands"]
          }
        }]
      }]
    });

    const response = result.response;
    const text = response.text();

    // Parse the JSON response
    const jsonResponse = JSON.parse(text);

    // Validate the response against our schema
    const validation = QaCommandSchema.safeParse(jsonResponse);
    if (!validation.success) {
      throw new Error(`Invalid AI response format: ${validation.error}`);
    }

    return jsonResponse;
  } catch (error) {
    core.setFailed(`AI analysis failed: ${error.message}`);
    throw error;
  }
}

async function executeQAChecks(commands) {
  // Validate schema structure
  const validation = QaCommandSchema.safeParse({ commands });
  if (!validation.success) {
    core.setFailed(`Invalid command structure: ${validation.error}`);
    return false;
  }

  for (const { name, aws_service, command, expected_output } of commands) {
    try {
      core.info(`Running AWS check: ${name} (${aws_service})`);

      // Execute AWS CLI command
      const output = execSync(command, { encoding: 'utf-8' });

      // Verify output using jq
      const verifyCommand = `echo '${JSON.stringify(output)}' | jq -r '${expected_output}'`;
      const verificationResult = execSync(verifyCommand, { encoding: 'utf-8' }).trim();

      if (!verificationResult || verificationResult === 'false') {
        core.setFailed(`AWS check failed: ${name}\nCommand: ${command}\nOutput: ${output}`);
        return false;
      }

      core.info(`✅ ${name} passed`);
    } catch (error) {
      core.setFailed(`AWS check error: ${name} - ${error.stderr || error.message}`);
      return false;
    }
  }
  return true;
}

async function run() {
  try {
    const tfApplyJob = core.getInput('tf_apply_job', { required: true });
    const geminiApiKey = core.getInput('gemini_api_key', { required: true });

    // Retrieve GitHub environment variables
    const token = process.env.GITHUB_TOKEN;
    const runId = process.env.GITHUB_RUN_ID;
    const repository = process.env.GITHUB_REPOSITORY;

    // Validate environment variables
    if (!token || !runId || !repository) {
      core.setFailed("Missing required environment variables (GITHUB_TOKEN, GITHUB_RUN_ID, GITHUB_REPOSITORY)");
      return;
    }

    // Initialize Octokit with the GitHub token
    const octokit = new Octokit({ auth: token });
    const [owner, repo] = repository.split('/');

    // Get all jobs in the current workflow run
    const { data: { jobs } } = await octokit.actions.listJobsForWorkflowRun({
      owner,
      repo,
      run_id: parseInt(runId),
    });

    // Find the target job by name
    const targetJob = jobs.find(job => job.name === tfApplyJob);
    if (!targetJob) {
      core.setFailed(`Job '${tfApplyJob}' not found in this workflow run`);
      return;
    }

    // Download the job logs
    const { data: logs } = await octokit.actions.downloadJobLogsForWorkflowRun({
      owner,
      repo,
      job_id: targetJob.id,
    });

    core.info(`Retrieved ${logs.length} characters of logs from job '${tfApplyJob}'`);

    // Analyze logs with Gemini
    const { commands } = await analyzeLogsWithAI(logs, geminiApiKey);
    core.info(`Generated ${commands.length} QA commands`);

    // Execute and verify commands
    const success = await executeQAChecks(commands);
    if (!success) {
      core.setFailed("One or more QA checks failed");
      return;
    }

    core.info("All QA checks passed successfully");
  } catch (error) {
    core.setFailed(`Action failed: ${error.message}`);
  }
}

run();