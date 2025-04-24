#!/bin/env python
# This script is used to parse a WDL and JSON file and check if any files are in GLACIER.
# If any files are in GLACIER, it will start the retrieval process.
import argparse
import logging

import boto3
import winval.cloud_files_validator as cfv

log_format = "[%(levelname)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format)


# parse input arguments
def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Parse JSON and WDL, check if any file in glacier, start retrieval if asked."
    )
    parser.add_argument("--wdl", type=str, required=True, help="The WDL to parse.")
    parser.add_argument("--param_json", type=str, required=True, help="The parameter file.")
    parser.add_argument("--retrieve", action="store_true", help="Retrieve the missing files from GLACIER.")
    parser.add_argument("--n_days", type=int, default=30, help="GLACIER retrieval time in days. Default is 90 days.")
    return parser.parse_args()


def main():
    args = parse_arguments()
    wdl = args.wdl
    param_json = args.param_json
    retrieve = args.retrieve

    # parse the WDL and JSON files
    cloud_validator = cfv.CloudFilesValidator(wdl, param_json)
    validation = cloud_validator.validate()
    if validation:
        logging.info("The WDL and JSON files are valid, no GLACIER  files found.")
    elif len(cloud_validator.non_validated_files) > len(cloud_validator.glacier_files):
        logging.info("Missing files found, correct first")
    elif len(cloud_validator.glacier_files) > 0:
        logging.info("The WDL and JSON files are valid, but the following files are in GLACIER:")
        for file in cloud_validator.glacier_files:
            logging.info(file)
        if retrieve:
            logging.info("Starting retrieval of the files from GLACIER.")
            s3 = boto3.client("s3")
            for file in cloud_validator.glacier_files:
                _, bucket, key = cfv.split_uri(file)
                _ = s3.restore_object(
                    Bucket=bucket,
                    Key=key,
                    RestoreRequest={
                        "Days": args.n_days,
                    },
                )
            logging.info("Retrieval started.")
        else:
            logging.info("Use --retrieve to start retrieval of the files from GLACIER.")


if __name__ == "__main__":
    main()
