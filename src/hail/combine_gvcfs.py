import hail as hl
import boto3

# Hail initialization
hl.init(
tmp_dir="s3://ultimagen-gil-hornung/hail/tmp/"
)

# List all UG gvcfs in my bucket,
s3 = boto3.client('s3')
bucket_name = 'ultimagen-gil-hornung'
prefix = 'gvcfs-for-hail/'
response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
gvcfs = [f"s3://{bucket_name}/{x['Key']}" for x in response['Contents'] if x['Key'].endswith('g.vcf.gz')]

combiner = hl.vds.new_combiner(output_path="s3://ultimagen-gil-hornung/combine_196.vds",
                               temp_path="s3://ultimagen-gil-hornung/hail/tmp/",
                               gvcf_paths=gvcfs,
                               use_genome_default_intervals=True,
                               gvcf_reference_entry_fields_to_keep=['GQ', 'MIN_DP'],
                               reference_genome='GRCh38'
                               )
combiner.run()
