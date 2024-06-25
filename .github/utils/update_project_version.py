import os
import re
import glob
import argparse
from re import sub

# Define the version pattern
version_pattern = re.compile(r'^(version\s*=\s*")[^"]*(".*)$', re.MULTILINE)


def main(version):
    version = version.lstrip('v')
    # Find all project.toml files
    project_files = glob.glob('**/pyproject.toml', recursive=True)

    for file_path in project_files:
        print(f"updating '{file_path}' version to {version}")
        with open(file_path, 'r+') as file:
            content = file.read()
            # Replace the version string
            new_version = f'version = "{version}"'
            new_content = sub(r'version[ ]+=[ ]+".*"', new_version, content)

            # Write the new content back to the file
            file.seek(0)
            file.write(new_content)
            file.truncate()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("version", help="Version to publish modules with", type=str)
    args = parser.parse_args()
    main(args.version)
