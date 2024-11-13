#!/bin/bash
# Get the package name from the first argument
package=$1

# Check if the package name is provided
if [ -z "$package" ]; then
  echo "Error: No package name provided."
  exit 1
fi

# Check if the package name is valid
if [ "$package" == "<PACKAGE>" ]; then
  echo "Error: Invalid package name <PACKAGE>. Please provide a package name in the format ugbio_<name>."
  exit 1
fi

# Update package list and install curl
apt-get update && apt-get install -y curl clang

# configure git
git config --global --add safe.directory /workspaces/ugbio-utils

# Set UV_LINK_MODE environment variable permanently
echo 'export UV_LINK_MODE=copy' >> ~/.bashrc
export UV_LINK_MODE=copy

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Add uv command to PATH
source $HOME/.local/bin/env

# Synchronize uv
uv sync --package $package
uv run pre-commit install

# Install VS Code extensions
echo "To install VS Code extensions, please run the following command:"
echo "cat /tmp/extensions.txt | xargs -L 1 code --install-extension"
