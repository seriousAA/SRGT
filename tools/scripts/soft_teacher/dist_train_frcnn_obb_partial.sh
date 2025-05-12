#!/usr/bin/env bash
set -x

CONFIG=configs/soft_teacher/softT_faster_rcnn_obb_r50_fpn_dota20_partial.py
work_dir=output/1_1_frcnn_softT_0614
resume_dir=output/1_1_frcnn_softT_0614/latest.pth

PYTHONPATH="$(dirname $0)/../../../":$PYTHONPATH

export unsup_start_iter=10000
export unsup_warmup_iter=5000

## uncomment the following command if you want to resume from the checkpoint in resume_dir
python -m torch.distributed.launch --nproc_per_node=1 --master_port=${PORT:-29515} \
    $(dirname "$0")/../../train.py $CONFIG --work-dir $work_dir --resume-from $resume_dir --launcher=pytorch \
        --cfg-options fold=1 percent=1 \
                    runner.max_iters=180000 \
                    lr_config.step=\[120000\]

## uncomment the following command if you want to train from scratch
# python -m torch.distributed.launch --nproc_per_node=1 --master_port=${PORT:-29515} \
#     $(dirname "$0")/../../train.py $CONFIG --work-dir $work_dir --launcher=pytorch \
#         --cfg-options fold=1 percent=1 \
#                     runner.max_iters=180000 \
#                     lr_config.step=\[120000\]
