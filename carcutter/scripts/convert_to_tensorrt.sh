#!/bin/bash
# Warning: This script has been vibe-coded with Claude 4.5 Sonnet

# --- Color Definitions ---
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

log_info()  { echo -e "${BLUE}[INFO]${NC} $@"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC} $@"; }
log_fail()  { echo -e "${RED}[FAIL]${NC} $@"; }
log_success()   { echo -e "${GREEN}[OK]${NC} $@"; }

if [ $# -ne 1 ]; then
    echo -e "${RED}Error: Usage: $0 <MODEL_REPOSITORY_FOLDER>${NC}"
    exit 1
fi

MODEL_REPOSITORY="${1}"
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1 | grep -oE '[A-Z][0-9]+' | head -n 1 | tr '[:upper:]' '[:lower:]')

if ! command -v trtexec &> /dev/null
then
    echo -e "${RED}trtexec not found, are you sure you are in the TensorRT container?${NC}"
    exit 1
fi

for model_dir in $MODEL_REPOSITORY/*/;
    do
        model_name=$(basename "$model_dir")
        echo "==============================="
        echo "Processing model: ${model_name}"
        echo "Model directory : ${model_dir}"

        # Find the latest version directory inside the model directory
        # Only consider subfolders with integer names as valid version directories
        model_version=$(find "${model_dir}" -mindepth 1 -maxdepth 1 -type d -printf "%f\n" | grep -E '^[0-9]+$' | sort -n -r | head -n 1)
        version_dir="${model_dir}${model_version}"

        echo "version_dir: ${version_dir}"
        echo "model.plan: ${model_dir}/model.plan"
        # Skip this model if a non-empty TensorRT plan file already exists for it
        if [ -n "$version_dir" ] && [ -f "${version_dir}/model.plan" ] && [ -s "${version_dir}/model.plan" ]; then
            log_info "[SKIP] Non-empty TensorRT plan already exists at ${version_dir}/model.plan, skipping ${model_name}."
            continue
        fi

        echo "Processing model: ${model_name} (dir: $version_dir) on GPU: $GPU"

        # Double-check the version directory (path sanity)
        # version_dir=$(find "${model_dir}" -mindepth 1 -maxdepth 1 -type d | head -n 1)
        if [ -z "$version_dir" ]; then
            log_warn "No version directory found in ${model_dir}, skipping ${model_name}."
        elif [ ! -f "${version_dir}/model.onnx" ]; then
            log_warn "No model.onnx found in $version_dir, skipping ${model_name}."
            continue
        else
            log_info "Converting ${model_name} (dir: $version_dir) to TensorRT plan..."
            param_args=""
            if [ -f "${model_dir}tensorrt_args.json" ]; then
                log_info "Found tensorrt_args.json for ${model_name}, loading custom trtexec parameters..."
                # If tensorrt_args.json exists, read custom trtexec arguments from it
                while IFS="=" read -r key value; do
                  # Clean up key (remove quotes and commas), trim spaces
                  key=$(echo "$key" | sed 's/[",]//g' | xargs)
                  # Clean up value (remove leading/trailing quotes), trim spaces
                  value=$(echo "$value" | sed 's/^"//' | sed 's/"$//' | xargs)
                  # Skip entries with missing keys or curly braces (JSON structure)
                  if [[ -z "$key" ]] || [[ "$key" == "{" ]] || [[ "$key" == "}" ]]; then
                    continue
                  fi
                  param_args="${param_args} --${key}=${value}"
                  log_info "[ARGS] Added parameter: --${key}=${value}"
                done < <(jq -r 'to_entries|map("\(.key)=\(.value)")|.[]' "${model_dir}tensorrt_args.json")
            fi
            GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1 | grep -oE '[A-Z][0-9]+' | head -n 1 | tr '[:upper:]' '[:lower:]')
            log_info "trtexec --onnx=\"${version_dir}/model.onnx\" --saveEngine=\"${version_dir}/model.plan\" --fp16 --verbose ${param_args}"
            trtexec --onnx="${version_dir}/model.onnx" --saveEngine="${version_dir}/model.plan" --fp16 --verbose ${param_args} &> "${version_dir}/model.trtexec.log"
            if [ $? -eq 0 ]; then
                log_success "Successfully converted ${model_name} to TensorRT plan: ${version_dir}/model.plan"
            else
                log_fail "Failed to convert ${model_name} (see log at ${version_dir}/model.trtexec.log)"
            fi
        fi

        config_file="${model_dir}config.pbtxt"
        # Update config.pbtxt to set platform to tensorrt_plan instead of onnxruntime_onnx
        if [ -f "$config_file" ]; then
            sed -i 's/onnxruntime_onnx/tensorrt_plan/g' "$config_file"
            log_info "Updated platform in $config_file to tensorrt_plan"
        fi
    done
