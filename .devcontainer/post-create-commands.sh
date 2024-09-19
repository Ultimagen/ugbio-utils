#!/bin/bash

# Update package list and install curl
apt-get update && apt-get install -y curl

# Set UV_LINK_MODE environment variable permanently
echo 'export UV_LINK_MODE=copy' >> ~/.bashrc
export UV_LINK_MODE=copy

# Install Rye
curl -sSf https://rye.astral.sh/get | RYE_INSTALL_OPTION="--yes" bash

# Add Rye to the shell environment
echo '. "$HOME/.rye/env"' >> ~/.bashrc && . ~/.bashrc

# Synchronize Rye
rye sync

# Install VS Code extensions
echo "To install VS Code extensions, please run the following command:"
echo "cat /tmp/extensions.txt | xargs -L 1 code --install-extension"