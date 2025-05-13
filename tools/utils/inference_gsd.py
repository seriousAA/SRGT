import torch
from ssod.apis import init_detector
from mmdet.apis import inference_detector
import mmcv

config_file = '/home/****/RS-PCT/configs/GSD/resnet50_fpn_gsd_estimation.py'
checkpoint_file = '/home/****/RS-PCT/output/gsd_estimation/0122/latest.pth'

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# test a single image and show the results
img = '/home/****/RS-PCT/data/DOTA/gsd_dataset/test/P0001_5.png'  # or img = mmcv.imread(img), which will only load it once
result = inference_detector(model, img)
pred_labels = torch.argmax(result).item()
print(pred_labels)