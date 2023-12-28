WORKFLOW_ID=$1

OMICS_COMPUTE_PRICING_SCRIPT="terraform/aws_ultimagen_pipelines/lambda/omics_end_run_handler/lambda/src/compute_pricing.py"

WORKFLOW=$(python $OMICS_COMPUTE_PRICING_SCRIPT $WORKFLOW_ID |  jq  ".info.runId")
echo "workflowId: $WORKFLOW"

TOTAL=$(python $OMICS_COMPUTE_PRICING_SCRIPT $WORKFLOW_ID |  jq  ".total")
echo "total cost: $TOTAL"

RUN_TIME=$(python $OMICS_COMPUTE_PRICING_SCRIPT $WORKFLOW_ID |  jq  ".cost_detail.storage_cost.run_duration_hr")
echo "total runtime: $RUN_TIME"

STORAGE=$(python $OMICS_COMPUTE_PRICING_SCRIPT $WORKFLOW_ID |  jq  ".cost_detail.storage_cost.cost")
echo "storage cost: $STORAGE"

MAKE_EXAMPLES_AVG=$(python $OMICS_COMPUTE_PRICING_SCRIPT $WORKFLOW_ID |  jq  ".cost_detail.task_costs" | jq '[ .[] | select( .name | contains("UGMakeExamples" )) | .duration_hr] | add / length')
echo "make examples time (avg): $MAKE_EXAMPLES_AVG"

MAKE_EXAMPLES_COST=$(python $OMICS_COMPUTE_PRICING_SCRIPT $WORKFLOW_ID |  jq  ".cost_detail.task_costs" | jq '[ .[] | select( .name | contains("UGMakeExamples" )) | .cost] | add')
echo "make examples cost: $MAKE_EXAMPLES_COST"

CALL_VARIANTS_TIME=$(python $OMICS_COMPUTE_PRICING_SCRIPT $WORKFLOW_ID |  jq  ".cost_detail.task_costs" | jq '[ .[] | select( .name | contains("UGCallVariants" )) | .duration_hr] | add')
echo "call variants time: $CALL_VARIANTS_TIME"

CALL_VARIANTS_COST=$(python $OMICS_COMPUTE_PRICING_SCRIPT $WORKFLOW_ID |  jq  ".cost_detail.task_costs" | jq '[ .[] | select( .name | contains("UGCallVariants" )) | .cost] | add')
echo "call variants cost: $CALL_VARIANTS_COST"

POST_PROCESS_TIME=$(python $OMICS_COMPUTE_PRICING_SCRIPT $WORKFLOW_ID |  jq  ".cost_detail.task_costs" | jq '[ .[] | select( .name | contains("UGPostProcessing" )) | .duration_hr] | add')
echo "post processing time: $POST_PROCESS_TIME"

POST_PROCESS_COST=$(python $OMICS_COMPUTE_PRICING_SCRIPT $WORKFLOW_ID |  jq  ".cost_detail.task_costs" | jq '[ .[] | select( .name | contains("UGPostProcessing" )) | .cost] | add')
echo "post processing cost: $POST_PROCESS_COST"


COMPRESS_GVCF_TIME=$(python $OMICS_COMPUTE_PRICING_SCRIPT $WORKFLOW_ID |  jq  ".cost_detail.task_costs" | jq '[ .[] | select( .name | contains("CompressGVCF" )) | .duration_hr] | add')
echo "compress gvcf time: $COMPRESS_GVCF_TIME"

COMPRESS_GVCF_COST=$(python $OMICS_COMPUTE_PRICING_SCRIPT $WORKFLOW_ID |  jq  ".cost_detail.task_costs" | jq '[ .[] | select( .name | contains("CompressGVCF" )) | .cost] | add')
echo "compress gvcf cost: $COMPRESS_GVCF_COST"
