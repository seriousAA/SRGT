# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import torch

from mmdet.models.builder import DETECTORS
from mmdet.utils import get_root_logger
from .obb_single_stage import OBBSingleStageDetector



@DETECTORS.register_module()
class OBBDinoDETR(OBBSingleStageDetector):
    """Implementation of rotated version 
    `DINO: DETR with Improved DeNoising Anchor Boxes for End-to-End Object Detection
    <https://arxiv.org/abs/2203.03605>`_ for OBB detection."""
    
    def __init__(self,
                 backbone,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        if init_cfg:
            self.reset_cls_fcs = init_cfg.pop('reset_cls_fcs', False)
        super(OBBDinoDETR, self).__init__(backbone, None, bbox_head, train_cfg,
                                   test_cfg, pretrained, init_cfg)
        if pretrained is not None:
            warnings.warn('DeprecationWarning: pretrained is a deprecated \
                key, please consider using init_cfg')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)

    def init_weights(self, pretrained=None):
        super(OBBDinoDETR, self).init_weights(pretrained)
        if hasattr(self, 'init_cfg') and self.init_cfg and self.reset_cls_fcs:
            logger = get_root_logger()
            logger.info('Reset the classifier weights of %s', self.bbox_head.__class__.__name__)
            if hasattr(self.bbox_head, 'reset_fc_cls_weights'):
                self.bbox_head.reset_fc_cls_weights()
            else:
                self.bbox_head.init_weights()
    
    # over-write `forward_dummy` because:
    # the forward of bbox_head requires img_metas
    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/analysis_tools/get_flops.py`
        """
        warnings.warn('Warning! MultiheadAttention in DETR does not '
                      'support flops computation! Do not use the '
                      'results in your papers!')

        batch_size, _, height, width = img.shape
        dummy_img_metas = [
            dict(
                batch_input_shape=(height, width),
                img_shape=(height, width, 3)) for _ in range(batch_size)
        ]
        x = self.extract_feat(img)
        outs = self.bbox_head(x, dummy_img_metas)
        return outs

