#!/usr/bin/env bash
set -x

CONFIG=configs/supervised_baseline/fcos_obb_r50_fpn_dota20_full.py
PYTHONPATH="$(dirname $0)/../../../":$PYTHONPATH

# Function to get current date in mmdd format
get_date() {
  date +%m%d
}

current_date=$(get_date)
output_folder=full_fcos_gsd/full_fcos_sup_only_${current_date}
work_dir=output/${output_folder}

export CUDA_VISIBLE_DEVICES=6

python $(dirname "$0")/../../train.py $CONFIG --work-dir $work_dir \
			--cfg-options evaluation.interval=2000000 \
          checkpoint_config.interval=20000 \
          checkpoint_config.max_keep_ckpts=100 \
          evaluation.non_cuda_parallel_merge=True


python $(dirname "$0")/../../test.py $CONFIG ${work_dir}/latest.pth \
	      --format-only --eval-options save_dir=results/${output_folder} \
          non_cuda_parallel_merge=True