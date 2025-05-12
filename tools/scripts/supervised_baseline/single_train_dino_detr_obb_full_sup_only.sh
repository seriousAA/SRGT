#!/usr/bin/env bash
set -x

CONFIG=configs/supervised_baseline/dino_detr_semi_head_r50_dota20_full.py
PYTHONPATH="$(dirname $0)/../../../":$PYTHONPATH

# Function to get current date in mmdd format
get_date() {
  date +%m%d
}

# Get the current date
current_date=$(get_date)
# Define work directory
output_folder=full_dino_detr_obb_sup_only_${current_date}
work_dir=output/${output_folder}
resume_dir=${work_dir}/latest.pth

## uncomment the following command if you want to train from scratch
CUDA_VISIBLE_DEVICES=2 \
	python $(dirname "$0")/../../train.py $CONFIG --work-dir $work_dir \
		--cfg-options runner.max_iters=180000 \
					lr_config.step=\[160000\] \
					evaluation.interval=20000 \
					evaluation.non_cuda_parallel_merge=True

# Proceed with the test command
CUDA_VISIBLE_DEVICES=2 \
	python $(dirname "$0")/../../test.py $CONFIG ${work_dir}/latest.pth \
		--format-only --eval-options save_dir=results/${output_folder} \
								non_cuda_parallel_merge=True