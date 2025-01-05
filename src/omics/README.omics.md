# ugbio_omics

This module includes Python scripts for working with AWS HealthOmics.

> Log in with SSH to the desired AWS account before running tools in this package.

List of tools:

1. **Compare Cromwell vs. Omics** - Process cost and performance on both platforms and save results into comparable CSV files. Also saves metadata and intermediate cost and performance files. Additionally, you can find a plots folder with a variety of HTML plots. Omics cost information is fetched from cost.csv in S3 run output bucket.

   Run `uv run compare_cromwell_omics --help` for more details.
2. **Compare Omics Runs** - Process cost and performance and compare multiple Omics runs. Omics cost information is fetched from cost.csv in S3 run output bucket.

   Run `uv run compare_omics_runs --help` for more details.
3. **Get Run Logs** - Download logs of an Omics run. For FAILED run by default, you'll get the logs of all failed tasks or the run's engine log if there are no failed tasks. For successful runs by default, you'll get all tasks' logs. You can use --task-id to get the log for a specific task.

   Run `uv run get_omics_logs --help` for more details.
4. **Get Omics Cache Path** - Get S3 uri of the cache of an Omics run. You can choose to copy the indexes files using --copy-indexes. Use --task-id to get the cache uri / copy indexes for a specific task.

   Run `uv run omics_cache_path --help` for more details.
5. **Manifest Log** - Download and parse the manifest log of an Omics run. The manifest log contains a lot of information about storage usage, CPU and memory usage per task, Docker images, inputs, and general information about the run.

6. **Performance** - Process CPU, memory, and I/O metrics from the monitor log running in each task. Get usage plots over time for a better understanding of your code utilization. Note that task logs must be accessible for this.

   Run `uv run performance --help` for more details.

Admin tools (requires admin AWS permissions):

1. Report Run Completion - Manually triggers the OmicsEndRunHandler AWS Lambda to report mongodb on end of omics run/s.
   Run `uv run report_run_completion --help` for more details.

## Run with Docker

A fast and easy way to run the above-mentioned tools is by running a container:

```sh
docker run -v ~/.aws:/root/.aws -e AWS_PROFILE=${AWS_PROFILE} -v <local_output>:<output> 337532070941.dkr.ecr.us-east-1.amazonaws.com/ugbio_omics:latest <tool>
```

Breakdown:

1. A simple `docker run` command.
2. `-v ~/.aws:/root/.aws -e AWS_PROFILE=${AWS_PROFILE}` - this will allow the Docker container to use the SSH profile you are logged into (customer/dev/prod).
   > Remember to use **alog** before running the container.

3. `-v <local_output>:<output>` - add a volume mapping and use it in the output path of the tool to get the outputs available outside the container.
4. `337532070941.dkr.ecr.us-east-1.amazonaws.com/ugbio_omics:latest` - Docker image.
5. `<tool>` - available tools are:
   * get_omics_logs
   * omics_cache_path
   * manifest_log
   * compare_omics_runs
   * compare_cromwell_omics
