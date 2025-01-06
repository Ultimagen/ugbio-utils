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
    for source_key in list_objects_with_index(s3_client, bucket_name, prefix):
        # Create destination key by removing '_index' substring
        dest_key = re.sub(r"_index", "", source_key)
        copy_object(s3_client, bucket_name, source_key, dest_key)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_id", help="AWS HealthOmics run id")
    parser.add_argument(
        "--task-id", type=str, help="HealthOmics run task-id. Leave empty to get the base run cache uri."
    )
    parser.add_argument("--copy-indexes", dest="copy_indexes", action="store_true", help="copy indexes cached files")
    parser.set_defaults(copy_indexes=False)
    args = parser.parse_args()

    cache_s3_uri = get_run_cache_path(args.run_id, args.task_id)
    if args.copy_indexes:
        copy_omics_cached_indexes(cache_s3_uri)


if __name__ == "__main__":
    main()
