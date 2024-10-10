# ugbio_omics
This module includes Python scripts for working with AWS HealthOmics.
> Log in with SSH to the desired AWS account before running tools in this package.

List of tools:

1. **Compare Cromwell vs. Omics** - Process cost and performance on both platforms and save results into comparable CSV files. Also saves metadata and intermediate cost and performance files. Additionally, you can find a plots folder with a variety of HTML plots. Omics cost information is collected using [Omics Run Analyzer](https://github.com/awslabs/amazon-omics-tools?tab=readme-ov-file#omics-run-analyzer).

    Run `uv run compare_cromwell_omics --help` for more details.

2. **Compare Omics Runs** - Process cost and performance and compare between multiple Omics runs. Omics cost information is collected using [Omics Run Analyzer](https://github.com/awslabs/amazon-omics-tools?tab=readme-ov-file#omics-run-analyzer).

    Run `uv run compare_omics_runs --help` for more details.

3. **Get Run Logs** - Download logs of an Omics run. You can download the logs for all tasks or for a specific task in the run.

    Run `uv run get_omics_logs --help` for more details.

4. **Manifest Log** - Download and parse the manifest log of an Omics run. The manifest log contains a lot of information about storage usage, CPU and memory usage per task, Docker images, inputs, and general information about the run.

    Run `uv run manifest_log --help` for more details.
