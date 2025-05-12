#!/usr/bin/env bash
set -x

# Function to get current date in mmdd format
get_date() {
  date +%m%d
}

current_date=$(get_date)
CONFIG=configs/srgt/softT_faster_rcnn_obb_r50_fpn_dota20_full.py
run_name=full_soft_srgt_${current_date}
output_folder=full_soft_srgt_gsd/${run_name}
work_dir=output/${output_folder}

PYTHONPATH="$(dirname $0)/../../../":$PYTHONPATH

export unsup_start_iter=40000
export unsup_warmup_iter=5000
export CUDA_VISIBLE_DEVICES=0

## uncomment the following command if you want to resume from the checkpoint in resume_dir
# python -m torch.distributed.launch --nproc_per_node=2 --master_port=${PORT:-29515} \
#     $(dirname "$0")/../../train.py $CONFIG --work-dir $work_dir --resume-from $resume_dir --launcher=pytorch \
#         --cfg-options fold=1 percent=1 \
#                     runner.max_iters=180000 \
#                     lr_config.step=\[120000\]

# uncomment the following command if you want to train from scratch
python $(dirname "$0")/../../train.py $CONFIG --work-dir $work_dir \
        --cfg-options evaluation.interval=2000000 \
          checkpoint_config.interval=20000 \
          checkpoint_config.max_keep_ckpts=100 \
          optimizer.lr=0.01 \
          semi_wrapper.train_cfg.rescale_pos_weight=False \
          semi_wrapper.train_cfg.no_unlabeled_gt=True \
          evaluation.non_cuda_parallel_merge=True \
          run_name=$run_name

python $(dirname "$0")/../../test.py $CONFIG ${work_dir}/latest.pth \
        --format-only --cfg-options run_name=test_$run_name work_dir=$work_dir runner.max_iters=720000 \
        --eval-options save_dir=results/${output_folder} \
          non_cuda_parallel_merge=True
