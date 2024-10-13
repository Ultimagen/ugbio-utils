#!/bin/bash

# Update package list and install curl
apt-get update && apt-get install -y curl clang

# Set UV_LINK_MODE environment variable permanently
echo 'export UV_LINK_MODE=copy' >> ~/.bashrc
export UV_LINK_MODE=copy

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Add uv command to PATH
source $HOME/.cargo/env

# Synchronize uv
uv sync

# Install VS Code extensions
echo "To install VS Code extensions, please run the following command:"
echo "cat /tmp/extensions.txt | xargs -L 1 code --install-extension"