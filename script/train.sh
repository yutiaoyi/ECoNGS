#!/usr/bin/env bash
set -e


cd "$(dirname "$0")/.."

rootDir=$1
TF_names=$2
exp_name=$3
lambda=$4

voxel_size=0.001
update_init_factor=4
appearance_dim=0

if [ -n "$TF_names" ]; then
    IFS=',' read -ra TF_ARRAY <<< "$TF_names"
    for i in "${!TF_ARRAY[@]}"; do
        TF_ARRAY[$i]=$(echo "${TF_ARRAY[$i]}" | xargs)
    done
    list="${TF_ARRAY[*]}"
    echo "Using TFs from parameter: ${list}"
else
    list=$(find "${rootDir}" -mindepth 1 -maxdepth 1 -type d -exec basename {} \; | sort)
    echo "TFs auto-discovered from rootDir: $list"
fi

tf_names=$(echo "${list}" | tr -s ' ' '_' | tr ' ' '_')

python train.py --eval -s "${rootDir}" \
                --tf_names "${tf_names}" \
                --voxel_size ${voxel_size} \
                --update_init_factor ${update_init_factor} \
                --appearance_dim ${appearance_dim} \
                --lmbda "${lambda}" \
                -m "../Exp/${exp_name}/${tf_names}"
