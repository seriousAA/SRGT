#!/usr/bin/env bash
set -x

CONFIG=configs/supervised_baseline/dino_detr_semi_head_r50_dota20_partial.py
PYTHONPATH="$(dirname $0)/../../../":$PYTHONPATH

# Function to get current date in mmdd format
get_date() {
  date +%m%d
}
DIRECTORY=output
data_root=data/DOTA/DOTA-v2.0_all

percent=2
# Define the ranges for the thresholds
iou_weights=$(seq 2.0 1.0 5.0)

for fold in {1..2}; do
    # Iterate over the rpn and cls thresholds
    for iou_weight in $iou_weights; do
        # Get the current date
        current_date=$(get_date)
        # Define work directory
        output_folder=${fold}_${percent}_detr_dino_sup_${iou_weight}_${current_date}
        work_dir=output/partial_detr_dino_sup_iou_weight/${output_folder}
        resume_dir=${work_dir}/latest.pth

        CUDA_VISIBLE_DEVICES=0 \
        python $(dirname "$0")/../../train.py $CONFIG --work-dir $work_dir \
            --cfg-options fold=$fold percent=$percent \
                        runner.max_iters=120000 \
                        model.bbox_head.loss_iou.loss_weight=$iou_weight \
                        model.train_cfg.assigner2.iou_cost.weight=$iou_weight \
                        evaluation.interval=20000
    done
done