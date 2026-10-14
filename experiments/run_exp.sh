#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: bash experiments/run_exp.sh NAME CONFIG PROCESSES_PER_GPU NUM_GPUS GPU_IDS EPISODES [SAVE_STEPS] [ARGS...]

EPISODES is "all", an episode label, or a subset name from constants.py.
SAVE_STEPS is true or false (default: false).
Set OPENAI_API_KEY for GPT models and DUMP_DIR to change the output directory.
Append --dry-run to inspect the command without starting evaluation.
EOF
}

if [[ ${1:-} == --help || ${1:-} == -h ]]; then
    usage
    exit 0
fi
if (( $# < 6 )); then
    usage >&2
    exit 2
fi

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
expname=$1
exp_config=$2
n_process_per_gpu=$3
num_gpu=$4
gpu_ids=$5
episode_labels=$6
save_step_data=${7:-false}
if [[ ! $expname =~ ^[[:alnum:]_.-]+$ || $expname == . || $expname == .. ]]; then
    echo "NAME must contain only letters, numbers, dots, underscores or hyphens." >&2
    exit 2
fi
if [[ $exp_config == */* || ! -f "$repo_dir/exp_configs/$exp_config" ]]; then
    echo "CONFIG must name an existing file in exp_configs/." >&2
    exit 2
fi
if [[ ! $n_process_per_gpu =~ ^[1-9][0-9]*$ || ! $num_gpu =~ ^[1-9][0-9]*$ ]]; then
    echo "PROCESSES_PER_GPU and NUM_GPUS must be positive integers." >&2
    exit 2
fi
if [[ ! $gpu_ids =~ ^[0-9]+(,[0-9]+)*$ ]]; then
    echo "GPU_IDS must be a comma-separated list of device indices." >&2
    exit 2
fi
IFS=, read -r -a devices <<< "$gpu_ids"
if (( ${#devices[@]} != num_gpu )); then
    echo "NUM_GPUS must match the number of entries in GPU_IDS." >&2
    exit 2
fi
if [[ $save_step_data != true && $save_step_data != false ]]; then
    echo "SAVE_STEPS must be true or false." >&2
    exit 2
fi

export CUDA_VISIBLE_DEVICES="$gpu_ids"
export MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet
export EXP_NAME="$expname"
dump_dir=${DUMP_DIR:-"$repo_dir/dump"}
cmd=(
    "${PYTHON:-python}" -u main_vis_multi.py
    -n "$n_process_per_gpu"
    --num_gpu "$num_gpu"
    --fmm_planner
    --edge_goal
    --temporal_collision
    --imagine_nav_planner
    --gt_scenegraph
    --exp_config "$exp_config"
    --group_caption_vlm gpt-4o-mini
    --scene_graph_prediction_llm gpt-4o-mini
    --ollama_port_start 14200
    --dump_location "$dump_dir/$expname"
)
if [[ $episode_labels != all ]]; then
    cmd+=(--episode_labels "$episode_labels")
fi
if [[ $save_step_data == true ]]; then
    cmd+=(--save_step_data)
fi
dry_run=false
for arg in "${@:8}"; do
    case "$arg" in
        --api|--api=*)
            echo "Set OPENAI_API_KEY instead of passing a key on the command line." >&2
            exit 2
            ;;
        --dry-run) dry_run=true ;;
        *) cmd+=("$arg") ;;
    esac
done

cd -- "$repo_dir"
if [[ $dry_run == true ]]; then
    printf '%q ' "${cmd[@]}"
    printf '\n'
    exit 0
fi
mkdir -p -- "$dump_dir/$expname/objectnav-dino"
exec > >(tee -a "$dump_dir/$expname/objectnav-dino/log.txt") 2>&1
exec "${cmd[@]}"
