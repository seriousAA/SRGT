#!/usr/bin/env bash
set -x

CONFIG=configs/supervised_baseline/fcos_obb_r50_fpn_dota20_partial.py
PYTHONPATH="$(dirname $0)/../../../":$PYTHONPATH

# Function to get current date in mmdd format
get_date() {
  date +%m%d
}

# Array of percent settings
percent_settings=(1 2 5 10)

# Loop over each percent setting
for percent in "${percent_settings[@]}"; do
	# Loop over each fold
	for fold in {2..10..2}; do
		# Get the current date
		current_date=$(get_date)
		# Define work directory
		output_folder=partial_fcos_obb_sup_only/${fold}_${percent}_fcos_sup_only_${current_date}
		work_dir=output/${output_folder}
		resume_dir=${work_dir}/latest.pth

		## uncomment the following command if you want to resume from the checkpoint in resume_dir
		# CUDA_VISIBLE_DEVICES=0 \
		# python $(dirname "$0")/../../train.py $CONFIG --work-dir $work_dir --resume-from $resume_dir \
		#     --cfg-options fold=$fold percent=$percent \
		# 				runner.max_iters=180000 \
		# 				lr_config.step=\[120000,160000\]

		## uncomment the following command if you want to train from scratch
		CUDA_VISIBLE_DEVICES=1 \
		python $(dirname "$0")/../../train.py $CONFIG --work-dir $work_dir \
			--cfg-options fold=$fold percent=$percent \
						runner.max_iters=180000 \
						lr_config.step=\[120000,160000\] \
						evaluation.non_cuda_parallel_merge=True \
                        evaluation.interval=20000 \

		# Define the variable that will hold the .pth file path
		pth_file="${work_dir}/latest.pth"

		# Branching logic to handle different cases based on $percent
		if [[ $percent -eq 1 || $percent -eq 2 ]]; then
			# Find all .pth files starting with "best_mAP_" in the work directory
			best_files=($(ls ${work_dir}/best_mAP_*.pth 2>/dev/null))

			if [[ ${#best_files[@]} -gt 0 ]]; then
				# Initialize variables for the highest iteration and corresponding file
				max_iter=0
				best_pth_file=""

				# Loop through each file and extract the iteration number
				for file in "${best_files[@]}"; do
					# Extract the iteration number using regex
					if [[ $file =~ best_mAP_iter_([0-9]+)\.pth ]]; then
						iter_num=${BASH_REMATCH[1]}
						# Compare and find the file with the highest iteration number
						if (( iter_num > max_iter )); then
							max_iter=$iter_num
							best_pth_file=$file
						fi
					fi
				done

				# If a valid file was found, set it as the .pth file to use
				if [[ -n $best_pth_file ]]; then
					pth_file=$best_pth_file
				fi
			fi
		fi

		# The pth_file variable now contains either the latest.pth or the best_mAP_*.pth file
		# Proceed with the test command
		CUDA_VISIBLE_DEVICES=1 \
		python $(dirname "$0")/../../test.py $CONFIG $pth_file \
			--format-only --eval-options save_dir=results/${output_folder} \
									non_cuda_parallel_merge=True
	done
done
