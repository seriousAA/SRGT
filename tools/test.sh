#!/usr/bin/env bash

CONFIG=configs/PseCo/PseCo_faster_rcnn_r50_caffe_fpn_coco_180k.py
CHECKPOINT=output/rssod/temp2/iter_25000.pth        # path to checkpoint

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python $(dirname "$0")/test.py $CONFIG $CHECKPOINT --eval bbox --out output/rssod/temp2/results.pkl
