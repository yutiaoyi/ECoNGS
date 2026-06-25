#!/usr/bin/env bash
set -e


cd "$(dirname "$0")/.."

model_dir=$1

python gui.py -so "${model_dir}"
