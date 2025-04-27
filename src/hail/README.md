### How to use this `cloudformation` tool

1. Change directories: `cd src/hail`
2. run `uv sync --package hail`
3. run `alog v1`
4. Using the text editor of your preference update the configuration file `config_EMR_spot.yaml` as per your request
5. run `sh cloudformation_hail_spot.sh`
6. Follow the instructions you see in the bottom of the cloudformation_hail_spot script log.
7. Monitoring cluster status: `aws emr describe-cluster --cluster-id <cluster-id> | jq ".Cluster" | jq ".Status"`
8. Terminate cluster: `aws emr terminate-clusters --cluster-id <cluster-id>`
