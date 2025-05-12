#!/usr/bin/env bash
set -x

CONFIG=configs/dense_teacher/sood_faster_rcnn_obb_r50_fpn_dota20_partial.py
PYTHONPATH="$(dirname $0)/../../../":$PYTHONPATH

# Function to get current date in mmdd format
get_date() {
  date +%m%d
}
rpn_pseudo_threshold=0.5
# Array of percent settings
percent_settings=(1 2 5 10)

# Loop over each percent setting
for percent in "${percent_settings[@]}"; do
	# Loop over each fold
	for fold in {1..10}; do
		# Get the current date
		current_date=$(get_date)
		# Define work directory
		output_folder=partial_sood_frcnn/${fold}_${percent}_sood_frcnn_${current_date}
		work_dir=output/${output_folder}
		resume_dir=${work_dir}/latest.pth

        export unsup_start_iter=10000
        export unsup_warmup_iter=5000

        ## uncomment the following command if you want to resume from the checkpoint in resume_dir
        # CUDA_VISIBLE_DEVICES=0 \
        # python $(dirname "$0")/../../train.py $CONFIG --work-dir $work_dir --resume-from $resume_dir \
        #     --cfg-options fold=1 percent=1 \
        #                 runner.max_iters=180000 \
        #                 lr_config.step=\[120000\]

		## uncomment the following command if you want to train from scratch
		CUDA_VISIBLE_DEVICES=0 \
		python $(dirname "$0")/../../train.py $CONFIG --work-dir $work_dir \
			--cfg-options fold=$fold percent=$percent \
                        semi_wrapper.train_cfg.rpn_pseudo_threshold=$rpn_pseudo_threshold \
						runner.max_iters=180000 \
						lr_config.step=\[120000\]

		## uncomment the following command if you want to test the model after the training
		CUDA_VISIBLE_DEVICES=0 \
		python $(dirname "$0")/../../test.py $CONFIG ${work_dir}/latest.pth \
			--format-only --eval-options save_dir=results/${output_folder}
    done
done