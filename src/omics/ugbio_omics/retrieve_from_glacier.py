#!/bin/env python
# This script is used to parse a WDL and JSON file and check if any files are in GLACIER.
# If any files are in GLACIER, it will start the retrieval process.
import argparse

import boto3
import winval.cloud_files_validator as cfv


# parse input arguments
def parse_arguments():
    parser = argparse.ArgumentParser(description="Parse JSON and WDL and (if necessary) start recovery from GLACIER.")
    parser.add_argument("--wdl", type=str, required=True, help="The WDL to parse.")
    parser.add_argument("--param_json", type=str, required=True, help="The parameter file.")
    parser.add_argument("--retrieve", action="store_true", help="Retrieve the missing files from GLACIER.")
    parser.add_argument("--n_days", type=int, default=90, help="GLACIER retrieval time in days. Default is 90 days.")
    return parser.parse_args()


def main():
    args = parse_arguments()
    wdl = args.wdl
    param_json = args.param_json
    retrieve = args.retrieve

    # parse the WDL and JSON files
    cloud_validator = cfv.CloudFilesValidator(wdl, param_json)
    validation = cloud_validator.validate_file()
    if validation:
        print("The WDL and JSON files are valid, no GLACIER  files found.")
    elif len(cloud_validator.non_validated_files) > len(cloud_validator.glacier_files):
        print("Missing files found, correct first")
    elif len(cloud_validator.glacier_files) > 0:
        print("The WDL and JSON files are valid, but the following files are in GLACIER:")
        for file in cloud_validator.glacier_files:
            print(file)
        if retrieve:
            print("Starting retrieval of the files from GLACIER.")
            s3 = boto3.client("s3")
            for file in cloud_validator.glacier_files:
                bucket, key = cfv.split_uri(file)
                _ = s3.restore_object(
                    Bucket=bucket,
                    Key=key,
                    RestoreRequest={
                        "Days": args.n_days,
                    },
                )
            print("Retrieval started.")
        else:
            print("Use --retrieve to start retrieval of the files from GLACIER.")


if __name__ == "__main__":
    main()
