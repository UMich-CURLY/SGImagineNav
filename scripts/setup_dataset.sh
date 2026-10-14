#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: bash scripts/setup_dataset.sh HM3D_SCENES_DIR OBJECTNAV_EPISODES_DIR

HM3D_SCENES_DIR contains hm3d_annotated_basis.scene_dataset_config.json and val/.
OBJECTNAV_EPISODES_DIR contains val/val.json.gz and val/content/.
The script links existing datasets into this checkout without deleting data.
EOF
}

if [[ ${1:-} == --help || ${1:-} == -h ]]; then
    usage
    exit 0
fi
if (( $# != 2 )); then
    usage >&2
    exit 2
fi
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ ! -f "$1/hm3d_annotated_basis.scene_dataset_config.json" || ! -d "$1/val" ]]; then
    echo "HM3D_SCENES_DIR must contain the annotated scene config and val/." >&2
    exit 2
fi
if [[ ! -f "$2/val/val.json.gz" || ! -d "$2/val/content" ]]; then
    echo "OBJECTNAV_EPISODES_DIR must contain val/val.json.gz and val/content/." >&2
    exit 2
fi
sources=("$(realpath -- "$1")" "$(realpath -- "$2")")
destinations=("$repo_dir/data/scene_datasets/hm3d" "$repo_dir/data/datasets/objectnav/hm3d/v1")
for index in 0 1; do
    destination=${destinations[$index]}
    if [[ -e $destination || -L $destination ]]; then
        if [[ $(realpath -- "$destination") != "${sources[$index]}" ]]; then
            echo "Dataset destination already exists: $destination" >&2
            exit 2
        fi
    fi
done
for index in 0 1; do
    destination=${destinations[$index]}
    if [[ ! -e $destination ]]; then
        mkdir -p -- "$(dirname -- "$destination")"
        ln -s -- "${sources[$index]}" "$destination"
    fi
    printf 'Linked %s\n' "$destination"
done
