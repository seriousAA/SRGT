import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models.builder import HEADS
from mmdet.models.roi_heads.bbox_heads import OBBConvFCBBoxHead
from mmdet.core import (multiclass_arb_nms, force_fp32, get_bbox_dim, bbox2type, OBBOverlaps)
from mmdet.ops.nms_rotated import arb_batched_nms
from mmdet.models.losses import accuracy

@HEADS.register_module()
class OBBConvFCBBoxHeadWithVar(OBBConvFCBBoxHead):
    def __init__(self, fc_out_channels=1024, *args, **kwargs):
        super(OBBConvFCBBoxHeadWithVar, self).__init__(
            num_shared_convs=0,
            num_shared_fcs=2,
            num_cls_convs=0,
            num_cls_fcs=0,
            num_reg_convs=0,
            num_reg_fcs=0,
            fc_out_channels=fc_out_channels,
            *args,
            **kwargs)

        if self.with_reg:
            # Add the additional Linear layer for bbox_pred_std
            out_dim_reg = self.reg_dim if self.reg_class_agnostic else self.reg_dim * self.num_classes
            self.fc_reg_var = nn.Linear(self.reg_last_dim, out_dim_reg)

    def init_weights(self):
        super(OBBConvFCBBoxHeadWithVar, self).init_weights()
        if self.with_reg:
            nn.init.normal_(self.fc_reg_var.weight, 0, 0.0001)
            nn.init.constant_(self.fc_reg_var.bias, 0)

    def forward(self, x):
        # Shared part
        if self.num_shared_convs > 0:
            for conv in self.shared_convs:
                x = conv(x)
        if self.num_shared_fcs > 0:
            if self.with_avg_pool:
                x = self.avg_pool(x)
            x = x.flatten(1)
            for fc in self.shared_fcs:
                x = self.relu(fc(x))

        # Separate branches
        x_cls = x
        x_reg = x

        for conv in self.cls_convs:
            x_cls = conv(x_cls)
        if x_cls.dim() > 2:
            if self.with_avg_pool:
                x_cls = self.avg_pool(x_cls)
            x_cls = x_cls.flatten(1)
        for fc in self.cls_fcs:
            x_cls = self.relu(fc(x_cls))

        for conv in self.reg_convs:
            x_reg = conv(x_reg)
        if x_reg.dim() > 2:
            if self.with_avg_pool:
                x_reg = self.avg_pool(x_reg)
            x_reg = x_reg.flatten(1)
        for fc in self.reg_fcs:
            x_reg = self.relu(fc(x_reg))

        # Predictions
        cls_score = self.fc_cls(x_cls) if self.with_cls else None
        bbox_pred = self.fc_reg(x_reg) if self.with_reg else None
        bbox_pred_std = self.fc_reg_var(x_reg) if self.with_reg else None

        return cls_score, bbox_pred, bbox_pred_std
    
    @force_fp32(apply_to=('cls_score', 'bbox_pred', 'bbox_pred_std'))
    def loss(self,
            cls_score,
            bbox_pred,
            bbox_pred_std,  # New parameter
            rois,
            labels,
            label_weights,
            bbox_targets,
            bbox_weights,
            gt_loc_std=None,  # Needed for pseudo-label loss
            reduction_override=None,
            **kwargs):
        losses = dict()

        # Classification loss (no major changes needed here)
        if cls_score is not None:
            avg_factor = max(torch.sum(label_weights > 0).float().item(), 1.)
            if cls_score.numel() > 0:
                losses['loss_cls'] = self.loss_cls(
                    cls_score,
                    labels,
                    label_weights,
                    avg_factor=avg_factor,
                    reduction_override=reduction_override)
                losses['acc'] = accuracy(cls_score, labels)

        # Bounding box regression loss
        if bbox_pred is not None:
            self.box_reg_loss_type = self.loss_bbox.__class__.__name__
            bg_class_ind = self.num_classes
            pos_inds = (labels >= 0) & (labels < bg_class_ind)

            # Regression for positive samples
            if pos_inds.any():
                target_dim = self.reg_dim
                if self.reg_decoded_bbox:
                    bbox_pred = self.bbox_coder.decode(rois[:, 1:], bbox_pred)
                    target_dim = get_bbox_dim(self.end_bbox_type)

                if self.reg_class_agnostic:
                    pos_bbox_pred = bbox_pred.view(bbox_pred.size(0), target_dim)[pos_inds.type(torch.bool)]
                    pos_bbox_pred_std = bbox_pred_std.view(bbox_pred.size(0), 
                                                           target_dim)[pos_inds.type(torch.bool)]  # New addition
                else:
                    pos_bbox_pred = bbox_pred.view(bbox_pred.size(0), -1, target_dim)[
                        pos_inds.type(torch.bool), labels[pos_inds.type(torch.bool)]]
                    pos_bbox_pred_std = bbox_pred_std.view(bbox_pred.size(0), -1, target_dim)[
                        pos_inds.type(torch.bool), labels[pos_inds.type(torch.bool)]]

                # Determine the type of loss to apply based on config (supervised vs unsupervised)
                if self.box_reg_loss_type == "SmoothL1Loss":
                    assert not self.reg_decoded_bbox
                    losses['loss_bbox'] = self.loss_bbox(
                        pos_bbox_pred,
                        bbox_targets[pos_inds.type(torch.bool)],
                        bbox_weights[pos_inds.type(torch.bool)],
                        avg_factor=bbox_targets.size(0),
                        reduction_override=reduction_override)

                elif self.box_reg_loss_type == "NLLoss":  # Custom loss incorporating std
                    assert not self.reg_decoded_bbox
                    pos_proposals = rois[pos_inds.type(torch.bool), 1:]
                    pos_bboxes = self.bbox_coder.decode(pos_proposals, pos_bbox_pred)
                    pos_bbox_targets = bbox_targets[pos_inds.type(torch.bool)]
                    pos_gt_bboxes = self.bbox_coder.decode(pos_proposals, pos_bbox_targets)
                    iou_calculator = OBBOverlaps()
                    IoUs = iou_calculator(
                        pos_bboxes,
                        pos_gt_bboxes, mode='iou',
                        is_aligned=True)
                    # Add logic for NLL-based loss (e.g., uncertainty-aware)
                    losses['loss_bbox'] = self.loss_bbox(
                        pos_bbox_pred,
                        pos_bbox_pred_std,
                        pos_bbox_targets,
                        iou_weight=IoUs.squeeze(),
                        avg_factor=bbox_targets.size(0),
                        reduction_override=reduction_override)

                elif self.box_reg_loss_type == "TSBetterPseudoLoss":
                    assert not self.reg_decoded_bbox and gt_loc_std is not None
                    # Add logic for pseudo-label loss using bbox_pred_std and gt_loc_std
                    losses['loss_bbox'] = self.loss_bbox(
                        pos_bbox_pred,
                        pos_bbox_pred_std,
                        bbox_targets[pos_inds.type(torch.bool)], 
                        gt_loc_std,
                        avg_factor=bbox_targets.size(0),
                        reduction_override=reduction_override)
                else:
                    losses['loss_bbox'] = self.loss_bbox(
                        pos_bbox_pred,
                        bbox_targets[pos_inds.type(torch.bool)],
                        bbox_weights[pos_inds.type(torch.bool)],
                        avg_factor=bbox_targets.size(0),
                        reduction_override=reduction_override)

            else:
                losses['loss_bbox'] = bbox_pred.sum() * 0  # No positive samples

        return losses

    @force_fp32(apply_to=('cls_score', 'bbox_pred', 'bbox_pred_std'))
    def get_bboxes(self,
                   rois,
                   cls_score,
                   bbox_pred,
                   img_shape,
                   scale_factor,
                   bbox_pred_std=None,
                   rescale=False,
                   cfg=None):
        if isinstance(cls_score, list):
            cls_score = sum(cls_score) / float(len(cls_score))
        scores = F.softmax(cls_score, dim=1) if cls_score is not None else None

        if bbox_pred is not None:
            bboxes = self.bbox_coder.decode(
                rois[:, 1:], bbox_pred, max_shape=img_shape)
        else:
            assert self.start_bbox_type == self.end_bbox_type
            bboxes = rois[:, 1:].clone()

        if rescale:
            if isinstance(scale_factor, float):
                scale_factor = [scale_factor for _ in range(4)]
            scale_factor = bboxes.new_tensor(scale_factor)

            bboxes = bboxes.view(bboxes.size(0), -1, get_bbox_dim(self.end_bbox_type))
            if self.end_bbox_type == 'hbb':
                bboxes /= scale_factor
            elif self.end_bbox_type == 'obb':
                bboxes[..., :4] = bboxes[..., :4] / scale_factor
            elif self.end_bbox_type == 'poly':
                bboxes /= scale_factor.repeat(2)
            bboxes = bboxes.view(bboxes.size(0), -1)

        if cfg is None:
            return bboxes, scores
        else:
            if bbox_pred_std is not None:
                det_bboxes, det_labels, det_bboxes_std = \
                    multiclass_arb_nms_with_var(bboxes, scores, 
                                                bbox_pred_std, cfg.score_thr,
                                                cfg.nms, cfg.max_per_img,
                                                bbox_type = self.end_bbox_type)
                return det_bboxes, det_labels, det_bboxes_std
            else:
                det_bboxes, det_labels = \
                    multiclass_arb_nms(bboxes, scores, cfg.score_thr,
                                       cfg.nms, cfg.max_per_img,
                                       bbox_type = self.end_bbox_type)
                return det_bboxes, det_labels


def multiclass_arb_nms_with_var(
        multi_bboxes,
        multi_scores,
        multi_bboxes_std,
        score_thr,
        nms_cfg,
        max_num=-1,
        score_factors=None,
        bbox_type='hbb'):
    
    bbox_dim = get_bbox_dim(bbox_type)
    num_classes = multi_scores.size(1) - 1
    # exclude background category
    if multi_bboxes.shape[1] > bbox_dim:
        bboxes = multi_bboxes.view(multi_scores.size(0), -1, bbox_dim)
        bboxes_std = multi_bboxes_std.view(multi_scores.size(0), -1, bbox_dim)
    else:
        bboxes = multi_bboxes[:, None].expand(-1, num_classes, bbox_dim)
        bboxes_std = multi_bboxes_std[:, None].expand(-1, num_classes, bbox_dim)
    scores = multi_scores[:, :-1]

    # filter out boxes with low scores
    valid_mask = scores > score_thr
    bboxes = bboxes[valid_mask]
    bboxes_std = bboxes_std[valid_mask]
    if score_factors is not None:
        scores = scores * score_factors[:, None]
    scores = scores[valid_mask]
    labels = valid_mask.nonzero()[:, 1]

    if bboxes.numel() == 0:
        bboxes = multi_bboxes.new_zeros((0, bbox_dim+1))
        bboxes_std = multi_bboxes_std.new_zeros((0, bbox_dim))
        labels = multi_bboxes.new_zeros((0, ), dtype=torch.long)
        return bboxes, labels, bboxes_std

    dets, keep = arb_batched_nms(bboxes, scores, labels, nms_cfg)

    if max_num > 0:
        dets = dets[:max_num]
        keep = keep[:max_num]

    return dets, labels[keep], bboxes_std[keep]
