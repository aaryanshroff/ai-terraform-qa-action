const core = require('@actions/core');
const { Octokit } = require("@octokit/rest");

async function run() {
  try {
    // Get input from action.yml
    const tfApplyJob = core.getInput('tf_apply_job', { required: true });

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

    // Use the logs in your QA checks
    core.info(`Retrieved ${logs.length} characters of logs from job '${tfApplyJob}'`);
  } catch (error) {
    core.setFailed(`Action failed: ${error.message}`);
  }
}

run();