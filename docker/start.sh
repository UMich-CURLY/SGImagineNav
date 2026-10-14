#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
compose=(docker compose -f "$repo_dir/docker/docker-compose.yaml")
if [[ -f "$repo_dir/.env" ]]; then
    compose+=(--env-file "$repo_dir/.env")
fi
exec "${compose[@]}" run --rm vln "$@"
