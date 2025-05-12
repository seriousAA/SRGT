#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022/9/18 21:01
# @Author : WeiHua
# @Time : 2024/07/01 22:24
# @Author : Yuqiu Li
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models import LOSSES, build_loss
from .utils.ot_tools import OT_Loss
from .rotated_semi_dt_loss import QFLv2

@LOSSES.register_module()
class RotatedTSOTDTLoss(nn.Module):
    def __init__(self, cls_channels=18, sample_num=512, dynamic_raw_type='ang', 
                 bbox_loss_type='l1', aux_loss=None, aux_loss_cfg=dict(), dynamic_fix_weight=None):
        """
        Symmetry Aware Two Stage Dense Teacher Loss.
        Args:
            cls_channels (int): number of classes
            sample_num (int): number of samples to use for loss calculation
            dynamic_raw_type (str): dynamic weighting strategy for loss
            aux_loss (str): auxiliary loss type to apply
            aux_loss_cfg (dict): configuration for the auxiliary loss
            dynamic_fix_weight (float): fixed weight for the dynamic loss
        """
        super(RotatedTSOTDTLoss, self).__init__()
        self.cls_channels = cls_channels
        self.sample_num = sample_num
        self.bbox_loss_type = bbox_loss_type
        if self.bbox_loss_type == 'l1':
            self.bbox_loss = nn.SmoothL1Loss(reduction='none')
        elif self.bbox_loss_type == 'iou':
            self.bbox_loss=build_loss(dict(type='RotatedIoULoss', reduction='none'))
        else:
            raise NotImplementedError(f"Unsupported bbox loss type: {self.bbox_loss_type}")
        if aux_loss:
            assert aux_loss in ['ot_loss_norm', 'ot_ang_loss_norm']
            self.ot_weight = aux_loss_cfg.pop('loss_weight', 1.)
            self.cost_type = aux_loss_cfg.pop('cost_type', 'all')
            assert self.cost_type in ['all', 'dist', 'score']
            self.clamp_ot = aux_loss_cfg.pop('clamp_ot', False)
            self.gc_loss = OT_Loss(**aux_loss_cfg)
        self.aux_loss = aux_loss
        self.apply_ot = self.aux_loss
        assert dynamic_raw_type in ['ang', '10ang', '50ang', '100ang']
        self.dynamic_raw_type = dynamic_raw_type
        if dynamic_fix_weight:
            self.dynamic_fix_weight = dynamic_fix_weight
        else:
            self.dynamic_fix_weight = 1.0

    def convert_shape(self, info):
        cls_scores = info["preds_cls"][:, :-1]
        centernesses = info["preds_cls"][:, -1, None]
        bbox_preds = info["preds_bbox"]
        assert len(cls_scores) == len(bbox_preds)

        return cls_scores, bbox_preds, centernesses

    def forward(self, teacher_info, student_info, num_per_img, **kwargs):
        """
        Args:
            teacher_info (dict): containing teacher's predictions
            student_info (dict): containing student's predictions
            num_per_img (list): number of predictions per image

        Returns:
            Dict: containing various computed losses
        """
        gpu_device = teacher_info["preds_cls"].device
        batch_size = len(num_per_img)

        t_cls_scores, t_bbox_preds, t_centernesses = self.convert_shape(teacher_info)
        s_cls_scores, s_bbox_preds, s_centernesses = self.convert_shape(student_info)

        with torch.no_grad():
            masks = []
            current_index = 0
            valid_num = 1e-6
            for num_logits in num_per_img:
                num_to_sample = min(num_logits, self.sample_num)
                teacher_probs = t_cls_scores[current_index:current_index+num_logits, :].sigmoid()
                max_vals = torch.max(teacher_probs, dim=1)[0]
                valid = max_vals >= 0.5
                sorted_vals, sorted_inds = torch.topk(max_vals, num_logits)
                indices = torch.zeros_like(max_vals)
                indices[sorted_inds[valid][:num_to_sample]] = 1
                valid_num += sorted_vals[valid][:num_to_sample].sum()
                masks.append(indices > 0)
                current_index += num_logits
            mask = torch.cat(masks)

        with torch.no_grad():
            if self.dynamic_raw_type in ['ang', '10ang', '50ang', '100ang']:
                # determine loss weights via angle difference
                loss_weight = torch.abs(t_bbox_preds[mask][:, -1] - s_bbox_preds[mask][:, -1]) / np.pi
                if self.dynamic_raw_type == '10ang':
                    loss_weight = torch.clamp(10 * loss_weight.unsqueeze(-1), 0, 1) + 1
                elif self.dynamic_raw_type == '50ang':
                    loss_weight = torch.clamp(50 * loss_weight.unsqueeze(-1), 0, 1) + 1
                elif self.dynamic_raw_type == '100ang':
                    loss_weight = torch.clamp(100 * loss_weight.unsqueeze(-1), 0, 1) + 1
                else:
                    loss_weight = loss_weight.unsqueeze(-1) + 1
            else:
                raise RuntimeError(f"Not support {self.dynamic_raw_type}")
        
        loss_cls = F.binary_cross_entropy(
            s_cls_scores[mask].sigmoid(),
            t_cls_scores[mask].sigmoid(),
            reduction="none",
        ).mean(dim=-1, keepdim=True)
        loss_cls = (loss_cls * loss_weight).sum() / valid_num
        
        # loss_cls = QFLv2(
        #     s_cls_scores.sigmoid(),
        #     t_cls_scores.sigmoid(),
        #     weight=mask.long(),
        #     reduction="sum",
        # ) / valid_num
        
        loss_bbox = self.bbox_loss(
            s_bbox_preds[mask],
            t_bbox_preds[mask],
        ) * t_centernesses[mask].sigmoid()
        loss_bbox = (loss_bbox * loss_weight).mean()
        loss_centerness = F.binary_cross_entropy(
            s_centernesses[mask].sigmoid(),
            t_centernesses[mask].sigmoid(),
            reduction='none'
        )
        loss_centerness = (loss_centerness * loss_weight).mean()
        unsup_losses = dict(loss_raw=self.dynamic_fix_weight * (loss_cls + loss_bbox + loss_centerness))

        # Compute auxiliary loss if specified
        if self.aux_loss:
            loss_gc = torch.zeros(1, device=gpu_device)
            aligned_proposals = kwargs.get('aligned_proposals', None)
            if aligned_proposals is None:
                return unsup_losses
            
            # Pre-process the score maps outside of the loop
            if self.aux_loss in ['ot_ang_loss_norm']:
                t_score_map = t_bbox_preds[:, -1, None]  # Use angle predictions directly
                s_score_map = s_bbox_preds[:, -1, None]
            else:
                t_score_map = t_cls_scores  # Use class scores directly
                s_score_map = s_cls_scores

            # Apply softmax or absolute processing based on aux_loss type
            if self.aux_loss in ['ot_loss_norm']:
                t_score_map = torch.softmax(t_score_map, dim=-1)
                s_score_map = torch.softmax(s_score_map, dim=-1)
            elif self.aux_loss in ['ot_ang_loss_norm']:
                t_score_map = torch.abs(t_score_map) / np.pi
                s_score_map = torch.abs(s_score_map) / np.pi

            start_idx = 0
            for i, num_logits in enumerate(num_per_img):
                end_idx = start_idx + num_logits
                map_mask = masks[i]
                num_to_sample = min(num_logits, self.sample_num)  # Reconfirm the number to sample
                assert map_mask.shape[0] == num_logits and map_mask.sum() <= num_to_sample
                
                # Extract the max class scores within each image's slice
                t_score, t_score_cls = torch.max(t_score_map[start_idx:end_idx][map_mask], dim=-1)
                s_score = s_score_map[start_idx:end_idx][map_mask][
                        torch.arange(t_score.shape[0], device=gpu_device),
                        t_score_cls]
                
                pts = aligned_proposals[i][map_mask][:, :2]  # Extract the coordinates of the sampled points

                if map_mask.sum() > 0:
                    # Calculate the OT-based loss for each batch
                    loss_gc += self.gc_loss(t_score, s_score, pts, cost_type=self.cost_type, clamp_ot=self.clamp_ot)
                start_idx = end_idx  # Move to the next batch of logits

            # Normalize the loss by the number of images
            unsup_losses.update(loss_gc=self.ot_weight * loss_gc)

        return unsup_losses
