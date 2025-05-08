#!/bin/bash

CLUSTER_ID=$1;
CORES=${2:-16};
CONNECTIONS=${3:-10000}
aws emr add-steps \
  --cluster-id "$CLUSTER_ID" \
  --steps "[
    {
      \"Type\": \"Spark\",
      \"Name\": \"Run Hail GVCF Combiner ${CORES} cores ${CONNECTIONS} connections\",
      \"ActionOnFailure\": \"CONTINUE\",
      \"Args\": [
        \"--conf\", \"spark.executor.cores=${CORES}\",
        \"--conf\", \"spark.hadoop.fs.s3.maxConnections=${CONNECTIONS}\",
        \"--conf\", \"spark.hadoop.fs.s3.socketTimeout=200000\",
        \"--conf\", \"spark.executor.heartbeatInterval=200000\",
        \"--conf\", \"spark.network.timeout=300000\",
        \"--conf\", \"spark.serializer=org.apache.spark.serializer.KryoSerializer\",
        \"--conf\", \"spark.kryo.registrator=is.hail.kryo.HailKryoRegistrator\",
        \"--conf\", \"spark.executor.memoryOverheadFactor=0.2\",
        \"--conf\", \"spark.executor.memory=100g\",
        \"--deploy-mode\", \"client\",
        \"--master\", \"yarn\",
        \"--jars\", \"s3://ultimagen-gil-hornung/hail/hail-all-spark.jar\",
        \"s3://ultimagen-gil-hornung/hail/combine_196_gvcfs.py\"
      ]
    }
  ]"