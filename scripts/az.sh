#!/usr/bin/env bash
# Use a host CLI when available; otherwise keep Azure CLI and credentials local.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if command -v az >/dev/null 2>&1; then
    exec az "$@"
fi
command -v docker >/dev/null || { echo "Docker oder Azure CLI installieren." >&2; exit 1; }
mkdir -p "$ROOT/.azure-local"
exec docker run --rm -i \
    -e AZURE_CONFIG_DIR=/azure-config \
    -v "$ROOT/.azure-local:/azure-config" \
    -v "$ROOT:/workspace" -w /workspace \
    mcr.microsoft.com/azure-cli:2.77.0 az "$@"
