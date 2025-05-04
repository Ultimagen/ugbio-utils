import hail as hl
import boto3
import uuid
from datetime import datetime

# Get today's date
today_date = datetime.today().strftime('%Y-%m-%d')

# Generate a UUID
unique_id = uuid.uuid4()

# Create the directory name
dir_name = f"{today_date}_{unique_id}"

temp_path=f"s3://ultimagen-gil-hornung/hail/tmp/"

# Hail initialization
hl.init(tmp_dir=temp_path)

# List all UG gvcfs in my bucket,
s3 = boto3.client('s3')
bucket_name = 'ultimagen-gil-hornung'
prefix = 'gvcfs-for-hail/'
response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
gvcfs = [f"s3a://{bucket_name}/{x['Key']}" for x in response['Contents'] if x['Key'].endswith('g.vcf.gz')]

combiner = hl.vds.new_combiner(output_path=f"s3://ultimagen-gil-hornung/hail/{dir_name}/combine_196.vds",
                               temp_path=temp_path,
                               gvcf_paths=gvcfs,
                               use_genome_default_intervals=True,
                               gvcf_reference_entry_fields_to_keep=['GQ', 'MIN_DP'],
                               reference_genome='GRCh38'
                               )
combiner.run()
