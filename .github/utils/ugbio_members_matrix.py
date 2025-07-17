#!/usr/bin/env python3
import argparse

# import json
# import os


def build_members_matrix(src_folder: str):
    # List directories in the src_folder that don't contain "__"
    # directories = [d for d in os.listdir(src_folder) if os.path.isdir(os.path.join(src_folder, d)) and "__" not in d]

    # print(json.dumps(directories, indent=4, sort_keys=True))
    print(["core"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-folder", required=False, help="Path to src folder", default="src")
    # parser.add_argument("--keep-internal", help="Keep internal ugbio members", action="store_true")

    args = parser.parse_args()

    build_members_matrix(args.src_folder)
