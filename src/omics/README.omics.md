# ugbio_omics

This module includes python scripts for working with AWS Healthe-Omics. 
> Login with ssh to the desried AWS account before running tools in this package.

List of tools:

1. **Compare cromwell vs. omics** - prcoess cost and performance on both platforms and save results into comprable csv. Also saves metadata and intermidaite cost and performance files. Additionally you can find a plots folder with varity of html plots. Omics cost information is collected using [omics run analyzer](https://github.com/awslabs/amazon-omics-tools?tab=readme-ov-file#omics-run-analyzer).

    Run `uv run compare_cromwell_omics --help` for more details.

2. **Compare omics runs** - prcoess cost and performance and compare between multiple omics runs. Omics cost information is collected using [omics run analyzer](https://github.com/awslabs/amazon-omics-tools?tab=readme-ov-file#omics-run-analyzer).

    Run `uv run compare_omics_runs --help` for more details.

3. **Get run logs** - download logs of omics run. You can donwaload the logs for all tasks or for a specifc task in the run.

    Run `uv run get_omics_logs --help` for more details.

4. **Manifest_log** - download and parse manifest log of omics run. The manifest log contains a lot of information about the storage usage, cpu and memory usage per task, docker images, inputs and general information of the run. 

    Run `uv run manifest_log --help` for more details.