CLUSTER_ID=$1;
CORES=$2-16;
CONNECTIONS=$3-2000;
aws emr add-steps --cluster-id $CLUSTER_ID --steps Type=Spark,Name="Run Hail GVCF Combiner 16 cores 2000 connections",ActionOnFailure=CONTINUE,Args='[
    "--conf","spark.executor.cores=${CORES}",
    "--conf","spark.hadoop.fs.s3.maxConnections=${CONNECTIONS}",
  "--deploy-mode","client",
  "--master","yarn",
  "--jars","s3://ultimagen-gil-hornung/hail/hail-all-spark.jar",
  "s3://ultimagen-gil-hornung/hail/combine_196_gvcfs.py"
]'