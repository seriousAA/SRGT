#!/usr/bin/env bash
set -x

CONFIG=configs/soft_teacher/softT_faster_rcnn_obb_r50_fpn_dota20_partial.py
PYTHONPATH="$(dirname $0)/../../../":$PYTHONPATH

# Function to get current date in mmdd format
get_date() {
  date +%m%d
}

# Function to find a GPU with at least 15G free memory
find_free_gpu() {
	local previous_gpu=$1
	while true; do
		available_gpus=($(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | awk '$1 >= 30000 {print NR-1}'))
		if [ ${#available_gpus[@]} -gt 0 ]; then
			# for gpu in 4 5; do
			# 	if [[ " ${available_gpus[@]} " =~ " ${gpu} " ]]; then
			# 		echo $gpu
			# 		return
			# 	fi
			# done
			if [[ " ${available_gpus[@]} " =~ " ${previous_gpu} " ]]; then
				echo $previous_gpu
				return
			fi
			echo ${available_gpus[0]}
			return
		fi
		echo "No GPU with at least 15G free memory found. Retrying in 30 minutes..."
		sleep 1800 # Wait for 30 minutes before retrying
	done
}

# Function to allow manual cancellation of sleep
manual_sleep() {
	local sleep_time=$1
	echo "Sleeping for $(($sleep_time / 3600)) hours (press Ctrl+C to cancel)..."

	trap 'echo "Sleep canceled by user."; return' SIGINT

	local end_time=$((SECONDS + sleep_time))
	while [ $SECONDS -lt $end_time ]; do
		remaining_time=$((end_time - SECONDS))
		if [ $remaining_time -gt 0 ]; then
			sleep 1
		else
			break
		fi
	done

	trap - SIGINT  # Reset the trap
}

rpn_pseudo_threshold=0.7
# Array of percent settings
percent_settings=(1 2 5 10)
free_gpu=5

# Loop over each percent setting
for percent in "${percent_settings[@]}"; do
	# Loop over each fold
	for fold in {2..10..2}; do
		# Get the current date
		current_date=$(get_date)
		# Define work directory
		output_folder=partial_softT_frcnn_focal_gsd/${fold}_${percent}_softT_frcnn_${current_date}
		work_dir=output/${output_folder}
		resume_dir=${work_dir}/latest.pth

    export unsup_start_iter=10000
    export unsup_warmup_iter=5000

    ## Uncomment the following command if you want to resume from the checkpoint in resume_dir
    # CUDA_VISIBLE_DEVICES=$free_gpu \
    # 	python $(dirname "$0")/../../train.py $CONFIG --work-dir $work_dir --resume-from $resume_dir \
    # 		--cfg-options fold=$fold percent=$percent \
    # 			runner.max_iters=180000 \
    # 			lr_config.step=\[160000\]

		## Uncomment the following command if you want to train from scratch
		CUDA_VISIBLE_DEVICES=$free_gpu \
		python $(dirname "$0")/../../train.py $CONFIG --work-dir $work_dir \
			--cfg-options fold=$fold percent=$percent \
            semi_wrapper.train_cfg.rpn_pseudo_threshold=$rpn_pseudo_threshold \
						runner.max_iters=180000 \
						lr_config.step=\[120000,160000\] \
						evaluation.interval=20000 \
						evaluation.non_cuda_parallel_merge=True
		wait

		## uncomment the following command if you want to test the model after the training
		CUDA_VISIBLE_DEVICES=$free_gpu \
		python $(dirname "$0")/../../test.py $CONFIG ${work_dir}/latest.pth \
			--format-only --eval-options save_dir=results/${output_folder} \
									non_cuda_parallel_merge=True
		wait
		set +x
			# Sleep for 2 hours with user interrupt option
			# echo "Sleeping for 2 hours before searching for the next available GPU..."
			manual_sleep 21600 # 2 hours in seconds

			# Find a GPU with at least 15G free memory
			echo "Searching for a GPU with at least 15G of free memory..."
			free_gpu=$(find_free_gpu $free_gpu)

			echo "Using GPU $free_gpu for the next loop."
		set -x
    done
done