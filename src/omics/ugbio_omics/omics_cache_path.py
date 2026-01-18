import argparse
import logging
import re
from urllib.parse import urlparse

import boto3

log_format = "[%(levelname)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format)


def build_cache_full_uri(s3_uri):
    parsed = urlparse(s3_uri)
    bucket = parsed.netloc
    prefix = parsed.path.lstrip("/")

    # Initialize S3 client
    s3_client = get_aws_client("s3")

    # Use Delimiter to list "subfolders"
    response = s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix, Delimiter="/")

    if "CommonPrefixes" not in response or len(response["CommonPrefixes"]) == 0:
        raise ValueError(f"No subfolders found under the prefix: {s3_uri}")
    if len(response["CommonPrefixes"]) > 1:
        raise ValueError(f"Multiple subfolders found under the prefix: {s3_uri}")

    # Retrieve the single subfolder
    subfolder = response["CommonPrefixes"][0]["Prefix"]
    # Construct and return the full S3 URI
    return f"s3://{bucket}/{subfolder}"


def get_run_cache_path(run_id, task_id=None):
    omics_client = get_aws_client("omics")

    omics_run = omics_client.get_run(id=run_id)
    logging.debug(f"omics run: {omics_run}")
    cache_id = omics_run["cacheId"]

    omics_cache = omics_client.get_run_cache(id=cache_id)
    logging.debug(f"omics run cache: {omics_cache}")
    cache_base_uri = f"{omics_cache['cacheS3Uri']}{run_id}"
    if not task_id:
        logging.info(f"Run cache uri: {cache_base_uri}")
        cache_uri = cache_base_uri
    else:
        task_cache_uri = f"{cache_base_uri}/{task_id}/"
        full_cache_uri = build_cache_full_uri(task_cache_uri)
        logging.info(f"Run task cache uri: {full_cache_uri}")
        cache_uri = full_cache_uri
    return cache_uri


def get_aws_client(service):
    return boto3.client(service)


def list_objects_with_index(s3_client, bucket_name, prefix):
    """
    List all objects under the given S3 prefix that contain '_index' in their keys.
    """
    paginator = s3_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if "_index" in key:
                yield key


def copy_object(s3_client, bucket_name, source_key, dest_key):
    """
    Copy an S3 object from source_key to dest_key within the same bucket.
    """
    copy_source = {"Bucket": bucket_name, "Key": source_key}
    s3_client.copy(copy_source, bucket_name, dest_key)
    logging.info(f"Copied {source_key} to s3://{bucket_name}/{dest_key}")


def process_s3_uri(s3_uri):
    """
    Parse the S3 URI and return the bucket name and prefix.
    """
    parsed = urlparse(s3_uri)
    if parsed.scheme != "s3":
        raise ValueError(f"Invalid S3 URI: {s3_uri}")
    return parsed.netloc, parsed.path.lstrip("/")


def copy_omics_cached_indexes(cache_s3_uri):
    # Initialize S3 client
    s3_client = get_aws_client("s3")

    # Parse bucket and prefix
    bucket_name, prefix = process_s3_uri(cache_s3_uri)

    # List and process objects
    found_index = False
    for source_key in list_objects_with_index(s3_client, bucket_name, prefix):
        found_index = True
        # Create destination key by removing '_index' substring
        dest_key = re.sub(r"_index", "", source_key)
        logging.info(f"Copying index file to: {dest_key}")
        copy_object(s3_client, bucket_name, source_key, dest_key)

    if not found_index:
        logging.warning(f"No index files found in cache path: {cache_s3_uri}")

def process_task_ids_file(run_id, task_ids_file, output_file, copy_indexes=False):
    """
    Process a file with task IDs and output a tab-separated file with task ID and cache path.
    If there's an error retrieving a cache path, leave the path blank but still output the line.
    If output_file is None, output to stdout.
    If copy_indexes is True, copy cached index files for each task.
    """
    import sys
    
    with open(task_ids_file, "r") as infile:
        outfile = sys.stdout if output_file is None else open(output_file, "w")
        try:
            # Write header
            outfile.write("task_id\tcache_path\n")
            
            for line in infile:
                task_id = line.strip()
                if not task_id:  # Skip empty lines
                    continue
                
                got_path = False
                try:
                    cache_path = get_run_cache_path(run_id, task_id)
                    outfile.write(f"{task_id}\t{cache_path}\n")
                    got_path = True
                except Exception as e:
                    logging.warning(f"Error retrieving cache path for task {task_id}: {e}")
                    outfile.write(f"{task_id}\t\n")

                if got_path and copy_indexes:
                    copy_omics_cached_indexes(cache_path)
        finally:
            if output_file is not None:
                outfile.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_id", help="AWS HealthOmics run id")
    parser.add_argument(
        "--task-id", type=str, help="HealthOmics run task ID. Leave empty to get the base run cache uri."
    )
    parser.add_argument(
        "--task-ids-file",
        type=str,
        help="File containing task IDs (one per line). Output will be written to a tab-separated file.",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Output file for task ID and cache path mapping (default: stdout)",
    )
    parser.add_argument("--copy-indexes", dest="copy_indexes", action="store_true", help="copy indexes cached files")
    parser.set_defaults(copy_indexes=False)
    args = parser.parse_args()

    if args.task_ids_file:
        # Process file with multiple task IDs
        process_task_ids_file(args.run_id, args.task_ids_file, args.output_file, args.copy_indexes)
    else:
        # Original single task ID behavior
        cache_s3_uri = get_run_cache_path(args.run_id, args.task_id)
        if args.copy_indexes:
            copy_omics_cached_indexes(cache_s3_uri)


if __name__ == "__main__":
    main()
