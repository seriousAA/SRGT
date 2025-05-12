import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models import LOSSES, build_loss

@LOSSES.register_module()
class RotatedMCLLoss(nn.Module):
    def __init__(self, 
                 cls_channels=18,
                 strides=[8, 16, 32, 64, 128],
                 coarse_sample_num=512,
                 fine_threshold=0.02,
                 bbox_loss_type='l1',
                 cls_loss_weight=1.0,
                 bbox_loss_weight=10.0,
                 centerness_loss_weight=10.0):
        super(RotatedMCLLoss, self).__init__()
        self.cls_channels = cls_channels
        self.strides = strides
        self.coarse_sample_num = coarse_sample_num
        self.fine_threshold = fine_threshold
        self.cls_loss_weight = cls_loss_weight
        self.bbox_loss_weight = bbox_loss_weight
        self.centerness_loss_weight = centerness_loss_weight
        self.bbox_loss_type = bbox_loss_type
        if self.bbox_loss_type == 'l1':
            self.bbox_loss = nn.SmoothL1Loss(reduction='none')
        elif self.bbox_loss_type == 'iou':
            self.bbox_loss=build_loss(dict(type='RotatedIoULoss', reduction='none'))
        else:
            raise NotImplementedError(f"Unsupported bbox loss type: {self.bbox_loss_type}")

    def reshape_predictions(self, preds):
        """Handle both single-level and multi-level predictions"""
        if isinstance(preds, tuple):  # Direct predictions
            cls_scores, bbox_preds, centerness = preds
            return cls_scores, bbox_preds, centerness
        elif isinstance(preds, list):  # Batch processing
            cls_scores = torch.cat([x[0] for x in preds], dim=0)
            bbox_preds = torch.cat([x[1] for x in preds], dim=0)
            centerness = torch.cat([x[2] for x in preds], dim=0)
            return cls_scores, bbox_preds, centerness
        else:
            raise TypeError("Invalid input type for predictions")

    def select_pseudo_labels(self, teacher_probs, teacher_centerness, valid_strides, batch_size):
        """Multi-level confidence-based pseudo label selection using strides"""
        # Ensure proper tensor shapes
        teacher_probs = teacher_probs.view(-1, self.cls_channels)  # [N, 18]
        teacher_centerness = teacher_centerness.view(-1)  # [N]
        
        # Group predictions by their feature levels
        level_masks = [valid_strides.view(-1) == s for s in self.strides]
        
        # Coarse selection for first two levels (P3 and P4 equivalent)
        coarse_inds = []
        for i in range(min(2, len(self.strides))):
            level_mask = level_masks[i]
            if level_mask.sum() == 0:
                continue
                
            level_probs = teacher_probs[level_mask]  # [M, 18]
            level_centerness = teacher_centerness[level_mask]  # [M]
            
            joint_conf = level_probs.max(dim=1)[0] * level_centerness  # [M]
            topk = min(self.coarse_sample_num*batch_size, len(joint_conf))
            coarse_inds.append(torch.nonzero(level_mask).squeeze(1)[torch.topk(joint_conf, topk)[1]])
        
        # Fine selection from all levels
        max_conf = teacher_probs.max(dim=1)[0]  # [N]
        joint_conf_all = max_conf * teacher_centerness  # [N]
        fine_inds = torch.nonzero(joint_conf_all > self.fine_threshold).squeeze(-1)
        
        # Combine selections
        selected_inds = torch.cat(coarse_inds + [fine_inds], 0) if coarse_inds else fine_inds
        selected_inds = selected_inds.unique()
        
        # Create weight mask
        weight_mask = torch.zeros_like(joint_conf_all)
        weight_mask[selected_inds] = joint_conf_all[selected_inds]
        
        return weight_mask, selected_inds, joint_conf_all

    def compute_losses(self, s_preds, t_preds, weight_mask, b_mask, max_joint_conf, loss_weight):
        """Compute classification, bbox and centerness losses"""
        s_cls, s_bbox, s_centerness = s_preds
        t_cls, t_bbox, t_centerness = t_preds

        if b_mask.sum() == 0:
            loss_cls = QFLv2(
                s_cls.sigmoid(),
                t_cls.sigmoid(),
                mask=max_joint_conf,
                weight=loss_weight,
                reduction="sum",
            ) / max_joint_conf.sum()

            return loss_cls, \
                    t_bbox.new_tensor(0.), \
                    t_centerness.new_tensor(0.)
        
        if loss_weight is not None:
            assert loss_weight.shape[0] == t_bbox.shape[0]
            if loss_weight.dim() == 1:
                loss_weight = loss_weight.unsqueeze(-1)
            else:
                assert loss_weight.dim() == 2 and loss_weight.shape[1] == 1

        # Classification loss (QFLv2)
        loss_cls = QFLv2(
            s_cls.sigmoid(),
            t_cls.sigmoid(),
            mask=weight_mask,
            weight=loss_weight,
            reduction="sum",
        ) / weight_mask.sum() * self.cls_loss_weight

        # Bbox regression loss
        loss_bbox = (self.bbox_loss(
            s_bbox[b_mask],
            t_bbox[b_mask]
        ) * weight_mask[b_mask].unsqueeze(-1)).mean() * self.bbox_loss_weight

        # Centerness loss
        loss_centerness = (F.binary_cross_entropy(
            s_centerness[b_mask].sigmoid(),
            t_centerness[b_mask].sigmoid(),
            reduction='none'
        ) * weight_mask[b_mask].unsqueeze(-1)).mean() * self.centerness_loss_weight

        return loss_cls, loss_bbox, loss_centerness

    def forward(self, teacher_info, student_info, num_per_img, loss_weight=None, valid_strides=None, **kwargs):
        """
        Args:
            teacher_info: Teacher predictions (tuple or list of tuples)
            student_info: Student predictions (tuple or list of tuples)
            num_per_img: Number of valid points per image
            loss_weight: Per-point loss weights
            valid_strides: Strides for each prediction point
        """
        # Reshape predictions
        t_cls, t_bbox, t_centerness = self.reshape_predictions(teacher_info)
        s_cls, s_bbox, s_centerness = self.reshape_predictions(student_info)
        
        # Handle valid_strides
        if valid_strides is None:
            raise ValueError("valid_strides must be provided")
        else:
            # Convert list of tensors to single tensor
            if isinstance(valid_strides, list):
                valid_strides = torch.cat(valid_strides)
        batch_size = len(num_per_img)
        with torch.no_grad():
            # Select pseudo labels
            weight_mask, b_mask, max_joint_conf = self.select_pseudo_labels(
                t_cls.sigmoid(), 
                t_centerness.sigmoid(),
                valid_strides,
                batch_size
            )
            b_mask = weight_mask > 0.
            
            # Apply loss weights if provided
            if loss_weight is not None:
                if isinstance(loss_weight, list):
                    loss_weight = torch.cat(loss_weight)

        # Compute losses
        loss_cls, loss_bbox, loss_centerness = self.compute_losses(
            (s_cls, s_bbox, s_centerness),
            (t_cls, t_bbox, t_centerness),
            weight_mask,
            b_mask,
            max_joint_conf,
            loss_weight
        )
        
        return {
            'loss_cls': loss_cls,
            'loss_bbox': loss_bbox,
            'loss_centerness': loss_centerness
        }


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
