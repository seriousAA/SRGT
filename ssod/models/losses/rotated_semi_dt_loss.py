#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022/9/18 21:01
# @Author : WeiHua
# @Time : 2024/07/01 22:24
# @Author : Yuqiu Li
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models import LOSSES, build_loss


@LOSSES.register_module()
class RotatedDTLoss(nn.Module):
    def __init__(self, cls_channels=18, sample_ratio=0.02, 
                 sample_num=None, cls_loss_type='qfl',
                 bbox_loss_type='l1'):
        super(RotatedDTLoss, self).__init__()
        self.cls_channels = cls_channels
        self.sample_num = max(1, sample_num) if sample_num else None
        self.sample_ratio = min(1.0, sample_ratio)
        self.cls_loss_type = cls_loss_type
        self.bbox_loss_type = bbox_loss_type
        if self.bbox_loss_type == 'l1':
            self.bbox_loss = nn.SmoothL1Loss(reduction='none')
        elif self.bbox_loss_type == 'iou':
            self.bbox_loss=build_loss(dict(type='RotatedIoULoss', reduction='none'))
        else:
            raise NotImplementedError(f"Unsupported bbox loss type: {self.bbox_loss_type}")

    def convert_shape(self, info):
        cls_scores = info["preds_cls"][:, :-1]
        centernesses = info["preds_cls"][:, -1, None]
        bbox_preds = info["preds_bbox"]
        assert len(cls_scores) == len(bbox_preds)

        return cls_scores, bbox_preds, centernesses

    def convert_shape_(self, infos):
        cls_scores = [info[0] for info in infos]
        bbox_preds = [info[1] for info in infos]
        centernesses = [info[2] for info in infos]
        
        cls_scores = torch.cat(cls_scores, dim=0)
        bbox_preds = torch.cat(bbox_preds, dim=0)
        centernesses = torch.cat(centernesses, dim=0)

        return cls_scores, bbox_preds, centernesses

    def concat_preds_if_batched(self, cls_scores, bbox_preds, centernesses):
        cls_scores = torch.cat(cls_scores, dim=0) if isinstance(cls_scores, list) else cls_scores
        bbox_preds = torch.cat(bbox_preds, dim=0) if isinstance(bbox_preds, list) else bbox_preds
        centernesses = torch.cat(centernesses, dim=0) if isinstance(centernesses, list) else centernesses
        return cls_scores, bbox_preds, centernesses

    def forward(self, teacher_info, student_info, num_per_img, loss_weight=None, **kwargs):
        """
        Description: Forward function for the RotatedDTLoss.
        Args: 
            teacher_info (dict): The teacher's prediction information.
            student_info (dict): The student's prediction information.
            num_per_img (list): The number of logits per image.
        Returns:
            dict: The unsupervised losses.
        """
        if isinstance(teacher_info, dict):
            t_cls_scores, t_bbox_preds, t_centernesses = self.convert_shape(teacher_info)
            s_cls_scores, s_bbox_preds, s_centernesses = self.convert_shape(student_info)
        elif isinstance(teacher_info, tuple):
            t_cls_scores, t_bbox_preds, t_centernesses = teacher_info
            s_cls_scores, s_bbox_preds, s_centernesses = student_info
        elif isinstance(teacher_info, list):
            t_cls_scores, t_bbox_preds, t_centernesses = self.convert_shape_(teacher_info)
            s_cls_scores, s_bbox_preds, s_centernesses = self.convert_shape_(student_info)
        else:
            raise RuntimeError("Invalid input type for teacher_info and student_info.")
        
        t_cls_scores, t_bbox_preds, t_centernesses = \
            self.concat_preds_if_batched(t_cls_scores, t_bbox_preds, t_centernesses)
        s_cls_scores, s_bbox_preds, s_centernesses = \
            self.concat_preds_if_batched(s_cls_scores, s_bbox_preds, s_centernesses)

        with torch.no_grad():
            mask = []
            current_index = 0
            fg_num = 1e-6
            for num_logits in num_per_img:
                if self.sample_num:
                    num_to_sample = min(num_logits, self.sample_num)
                else:
                    num_to_sample = max(1, int(num_logits * self.sample_ratio))

                teacher_probs = \
                    t_cls_scores[current_index:current_index + num_logits, :].sigmoid()
                max_vals = torch.max(teacher_probs, dim=1)[0]
                sorted_vals, sorted_inds = torch.topk(max_vals, num_logits)
                indices = torch.zeros_like(max_vals)
                indices[sorted_inds[:num_to_sample]] = 1
                fg_num += sorted_vals[:num_to_sample].sum()
                mask.append(indices)
                current_index += num_logits
            mask = torch.cat(mask)
            b_mask = mask > 0

        if b_mask.sum() == 0:
            return dict(loss_cls=t_cls_scores.new_tensor(0.),
                        loss_bbox=t_bbox_preds.new_tensor(0.),
                        loss_centerness=t_centernesses.new_tensor(0.))
        
        if loss_weight is not None:
            assert loss_weight.shape[0] == t_bbox_preds.shape[0]
            if loss_weight.dim() == 1:
                loss_weight = loss_weight.unsqueeze(-1)
            else:
                assert loss_weight.dim() == 2 and loss_weight.shape[1] == 1
        
        if self.cls_loss_type == 'qfl':
            loss_cls = QFLv2(
                s_cls_scores.sigmoid(),
                t_cls_scores.sigmoid(),
                mask=mask,
                weight=loss_weight,
                reduction="sum",
            ) / fg_num
        else:
            loss_cls = F.binary_cross_entropy(
                s_cls_scores.sigmoid(),
                t_cls_scores.sigmoid(),
                reduction="none",
            )
            loss_cls = (loss_cls * loss_weight 
                        if loss_weight is not None else loss_cls).mean()
        
        loss_bbox = self.bbox_loss(
            s_bbox_preds[b_mask],
            t_bbox_preds[b_mask],
        ) * t_centernesses.sigmoid()[b_mask]
        loss_bbox = (loss_bbox * loss_weight[b_mask] 
                     if loss_weight is not None else loss_bbox).mean()
        
        loss_centerness = F.binary_cross_entropy(
            s_centernesses[b_mask].sigmoid(),
            t_centernesses[b_mask].sigmoid(),
            reduction='none'
        )
        loss_centerness = (loss_centerness * loss_weight[b_mask]
                            if loss_weight is not None else loss_centerness).mean()

        unsup_losses = dict(
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            loss_centerness=loss_centerness
        )

        return unsup_losses


def QFLv2(pred_sigmoid,
          teacher_sigmoid,
          mask,
          weight=None,
          beta=2.0,
          reduction='mean'):
    """
    Description: This is an implementation of the Quality Focal Loss v2.
    Args:
        pred_sigmoid (Tensor): The prediction with sigmoid.
        teacher_sigmoid (Tensor): The target with sigmoid.
        mask (Tensor): The mask for the loss.
        weight (Tensor): The weight of each sample.
        beta (float): The beta parameter for Focal Loss.
        reduction (str): The method used to reduce the loss.
    Returns:
        Tensor: The loss.
    """
    # all goes to 0
    pt = pred_sigmoid
    zerolabel = pt.new_zeros(pt.shape)
    loss = F.binary_cross_entropy(
        pred_sigmoid, zerolabel, reduction='none') * pt.pow(beta)
    pos = mask > 0

    # positive goes to bbox quality
    pt = teacher_sigmoid[pos] - pred_sigmoid[pos]
    loss[pos] = F.binary_cross_entropy(
        pred_sigmoid[pos], teacher_sigmoid[pos], reduction='none') * pt.pow(beta)

    if weight is not None:
        loss = loss * weight
    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()
    return loss
