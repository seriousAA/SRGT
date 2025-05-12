#!/usr/bin/env bash
set -x

# Function to get current date in mmdd format
get_date() {
  date +%m%d
}

current_date=$(get_date)
CONFIG=configs/supervised_baseline/faster_rcnn_obb_r50_fpn_dota20_full.py
BASE_DIR=full_frcnn_sup_only

# Define the iterations to train with corresponding lr steps
# Base setting: max_iters=180000, lr_steps=[120000, 160000]
declare -A training_configs=(
  ["180000"]="120000,160000"
  ["360000"]="240000,320000"
  ["720000"]="480000,640000"
)

# Export PYTHONPATH
PYTHONPATH="$(dirname \$0)/../../../":$PYTHONPATH

# Set visible GPUs
export CUDA_VISIBLE_DEVICES=0,1

# Loop through different iteration settings
for max_iters in "${!training_configs[@]}"; do
  # Get the lr steps for current iteration setting
  IFS=',' read -r step1 step2 <<< "${training_configs[$max_iters]}"
  
  # Set output folder
  output_folder=${BASE_DIR}/full_frcnn_sup_$((max_iters/1000))k_${current_date}
  work_dir=output/${output_folder}
  
  # Train with the current iteration setting
  python -m torch.distributed.launch --nproc_per_node=2 --master_port=${PORT:-29515} \
      $(dirname "$0")/../../train.py $CONFIG --work-dir $work_dir --launcher=pytorch \
          --cfg-options optimizer.lr=$(printf "%.2f" $(echo "0.01*2" | bc)) \
            data.samples_per_gpu=3 \
            evaluation.interval=$(((max_iters/2)/9)) \
            evaluation.non_cuda_parallel_merge=True \
            lr_config.step="[$((step1/2)), $((step2/2))]" \
            runner.max_iters=$((max_iters/2))

  # Test the trained model
  python -m torch.distributed.launch --nproc_per_node=2 --master_port=${PORT:-29515} \
      $(dirname "$0")/../../test.py $CONFIG ${work_dir}/latest.pth --launcher=pytorch \
          --format-only --eval-options save_dir=results/${output_folder} \
            non_cuda_parallel_merge=True
done
