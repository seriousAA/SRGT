#!/usr/bin/env bash
set -x

CONFIG=configs/soft_teacher/softT_faster_rcnn_obb_r50_fpn_dota20_partial.py
PYTHONPATH="$(dirname $0)/../../../":$PYTHONPATH

# Function to get current date in mmdd format
get_date() {
  date +%m%d
}
DIRECTORY=output
data_root=data/DOTA/DOTA-v2.0_all

percent=2
# Define the ranges for the thresholds
rpn_pseudo_threshold=0.7
unsup_weights=$(seq 0.5 0.5 3.0)

for fold in {1..2}; do
    # Iterate over the rpn and cls thresholds
    for unsup_weight in $unsup_weights; do
        # Get the current date
        current_date=$(get_date)
        # Define work directory
        output_folder=${fold}_${percent}_frcnn_softT_${unsup_weight}_${current_date}
        work_dir=output/partial_frcnn_softT_focal_ablation/${output_folder}
        resume_dir=${work_dir}/latest.pth

        export unsup_start_iter=10000
        export unsup_warmup_iter=5000

        CUDA_VISIBLE_DEVICES=0 \
        python $(dirname "$0")/../../train.py $CONFIG --work-dir $work_dir \
            --cfg-options fold=$fold percent=$percent \
                        runner.max_iters=100000 \
                        semi_wrapper.train_cfg.rpn_pseudo_threshold=$rpn_pseudo_threshold \
                        semi_wrapper.train_cfg.unsup_weight=$unsup_weight \
                        evaluation.non_cuda_parallel_merge=True \
                        evaluation.interval=20000

        CUDA_VISIBLE_DEVICES=0 \
        python $(dirname "$0")/../../test.py $CONFIG ${work_dir}/latest.pth \
            --cfg-options \
                data.test.ann_file=${data_root}/trainval/semi_supervised/${fold}_${percent}/unlabeled \
                data.test.img_prefix=${data_root}/trainval/images \
            --eval mAP --save-log \
            --eval-options with_merge=false
    done
done