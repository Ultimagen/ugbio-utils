import re
import tomllib
import tomli_w
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
        with open(file_path, 'rb') as file:
            data = tomllib.load(file)
            # Replace the version string
        data['project']['version'] = version
        with open(file_path, 'wb') as file:
            tomli_w.dump(data, file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("version", help="Version to publish modules with", type=str)
    args = parser.parse_args()
    main(args.version)
