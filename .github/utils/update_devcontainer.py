import argparse
import logging
import os
import re

import json5

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)
ECR_REPO = "337532070941.dkr.ecr.us-east-1.amazonaws.com"
DOCKER_PREFIX = "ugbio"
IMAGE_REGEX = r'^337532070941\.dkr\.ecr\.us-east-1\.amazonaws\.com/ugbio_[^:]+:[^"]+$'


def is_ugbio_image_used_in_file(file_path):
    with open(file_path) as f:
        data = json5.load(f)

    # Check if the JSON has the "image" field
    if "image" in data:
        # Search for the image tag in the file
        image_match = re.match(IMAGE_REGEX, data["image"])
        return image_match
    return False


def update_image_in_file(file_path, member, new_version):
    new_image = f"{ECR_REPO}/{DOCKER_PREFIX}_{member}:{new_version}"
    updated_lines = []

    with open(file_path) as f:
        lines = f.readlines()

    for line in lines:
        stripped_line = line.strip()
        if stripped_line.startswith('"image"'):
            # Update the line containing the image key
            updated_line = f'	"image": "{new_image}",\n'
            updated_lines.append(updated_line)
            log.info(f"Updating {file_path}: {new_image}")
        else:
            # Keep the line as it is
            updated_lines.append(line)

    # Write the updated lines back to the file without changing formatting or comments
    with open(file_path, "w") as f:
        f.writelines(updated_lines)


def update_devcontainers(repo_path, new_version):
    # Walk through the repository to find all devcontainer.json files
    for root, _, files in os.walk(repo_path):
        for file in files:
            if file == "devcontainer.json":
                file_path = os.path.join(root, file)
                # The member is assumed to be the immediate subfolder name
                member = os.path.basename(root)
                if is_ugbio_image_used_in_file(file_path):
                    update_image_in_file(file_path, member, new_version)
                else:
                    log.info(f"No image field/matching image found in {file_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Update version in devcontainer.json files.")
    parser.add_argument("--version", required=True, help="New version to update to (e.g., 1.2.0)")
    parser.add_argument(
        "--base-path", required=False, help="Path to the .devcontainer root folder", default=".devcontainer"
    )

    args = parser.parse_args()

    update_devcontainers(args.base_path, args.version)
