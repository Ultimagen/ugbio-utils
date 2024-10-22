import argparse
import glob
import tomllib

import tomli_w


def _update_toml_version(glob_regex, project_key_name, version):
    toml_files = glob.glob(glob_regex, recursive=True)

    for file_path in toml_files:
        print(f"updating '{file_path}' version to {version}")
        with open(file_path, "rb") as file:
            data = tomllib.load(file)
            # Replace the version string
        data[project_key_name]["version"] = version
        with open(file_path, "wb") as file:
            tomli_w.dump(data, file)


def main(version):
    version = version.lstrip("v")
    # Find and update all pyproject.toml files
    _update_toml_version("**/pyproject.toml", "project", version)
    # Find and update all Cargo.toml files
    _update_toml_version("**/Cargo.toml", "package", version)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("version", help="Version to publish modules with", type=str)
    args = parser.parse_args()
    main(args.version)
