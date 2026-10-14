#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec docker build -t sg-vln:hm3d -f "$repo_dir/docker/Dockerfile" "$repo_dir" "$@"
