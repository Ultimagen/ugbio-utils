#!/usr/bin/env bash

set -euo pipefail
IFS=$'\n\t'

LOG_GROUP_NAME="/aws/omics/WorkflowLog"
LOG_STREAM_NAME=$1
REGION="us-east-1"
OUTPUT_FILE="${LOG_STREAM_NAME//\//_}.log"

echo ${OUTPUT_FILE}
result=$(aws logs get-log-events \
    --output json \
    --start-from-head \
    --log-group-name=${LOG_GROUP_NAME} \
    --log-stream-name=${LOG_STREAM_NAME} \
    --region=${REGION})

echo ${result} | jq -r .events[].message >> ${OUTPUT_FILE}

nextToken=$(echo $result | jq -r .nextForwardToken)
while [ -n "$nextToken" ]; do
    echo ${nextToken}
    result=$(aws logs get-log-events \
      --output json \
      --start-from-head \
      --log-group-name=${LOG_GROUP_NAME} \
      --log-stream-name=${LOG_STREAM_NAME} \
      --region=${REGION} \
      --next-token="${nextToken}")

    if [[ $(echo ${result} | jq -e '.events == []') == "true" ]]; then
        echo "response with empty events found -> exiting."
        exit
    fi

    echo ${result} | jq -r .events[].message >> ${OUTPUT_FILE}

    nextToken=$(echo ${result} | jq -r .nextForwardToken)
done
