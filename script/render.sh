#!/usr/bin/env bash
set -e


cd "$(dirname "$0")/.."

view_path=$1
model_dir=$2
output_dir=$3
TF_names=$4

python multi_render.py -s "${view_path}" \
                       --root_model_dir "${model_dir}" \
                       --output_dir "${output_dir}" \
                       --TF_names ${TF_names}
