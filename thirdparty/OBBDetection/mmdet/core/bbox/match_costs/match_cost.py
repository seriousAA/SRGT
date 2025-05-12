# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, Union
from ..iou_calculators import bbox_overlaps
from ..transforms import bbox_cxcywh_to_xyxy, bbox_xyxy_to_cxcywh
from ..transforms_obb import bbox2type
from .builder import MATCH_COST
from mmdet.ops import obb_overlaps


@MATCH_COST.register_module()
class BBoxL1Cost:
    """BBoxL1Cost.

     Args:
         weight (int | float, optional): loss_weight
         box_format (str, optional): 'xyxy' for DETR, 'xywh' for Sparse_RCNN

     Examples:
         >>> from mmdet.core.bbox.match_costs.match_cost import BBoxL1Cost
         >>> import torch
         >>> self = BBoxL1Cost()
         >>> bbox_pred = torch.rand(1, 4)
         >>> gt_bboxes= torch.FloatTensor([[0, 0, 2, 4], [1, 2, 3, 4]])
         >>> factor = torch.tensor([10, 8, 10, 8])
         >>> self(bbox_pred, gt_bboxes, factor)
         tensor([[1.6172, 1.6422]])
    """

    def __init__(self, weight=1., box_format='xyxy'):
        self.weight = weight
        assert box_format in ['xyxy', 'xywh', 'obb']
        self.box_format = box_format

    def __call__(self, bbox_pred, gt_bboxes, box_format=None):
        """
        Args:
            bbox_pred (Tensor): Predicted boxes with normalized coordinates
                (cx, cy, w, h), which are all in range [0, 1]. Shape
                [num_query, 4].
            gt_bboxes (Tensor): Ground truth boxes with normalized
                coordinates (x1, y1, x2, y2). Shape [num_gt, 4].

        Returns:
            torch.Tensor: bbox_cost value with weight
        """
        if box_format is not None:
            self.box_format = box_format
        if self.box_format == 'xywh':
            gt_bboxes = bbox_xyxy_to_cxcywh(gt_bboxes)
        elif self.box_format == 'xyxy':
            bbox_pred = bbox_cxcywh_to_xyxy(bbox_pred)
        elif self.box_format == 'obb':
            gt_bboxes = bbox2type(gt_bboxes, 'obb')
            bbox_pred = bbox2type(bbox_pred, 'obb')
        assert gt_bboxes.shape[-1] == bbox_pred.shape[-1]
        
        bbox_cost = torch.cdist(bbox_pred, gt_bboxes, p=1)
        return bbox_cost * self.weight


@MATCH_COST.register_module()
class FocalLossCost:
    """FocalLossCost.

     Args:
         weight (int | float, optional): loss_weight
         alpha (int | float, optional): focal_loss alpha
         gamma (int | float, optional): focal_loss gamma
         eps (float, optional): default 1e-12

     Examples:
         >>> from mmdet.core.bbox.match_costs.match_cost import FocalLossCost
         >>> import torch
         >>> self = FocalLossCost()
         >>> cls_pred = torch.rand(4, 3)
         >>> gt_labels = torch.tensor([0, 1, 2])
         >>> factor = torch.tensor([10, 8, 10, 8])
         >>> self(cls_pred, gt_labels)
         tensor([[-0.3236, -0.3364, -0.2699],
                [-0.3439, -0.3209, -0.4807],
                [-0.4099, -0.3795, -0.2929],
                [-0.1950, -0.1207, -0.2626]])
    """

    def __init__(self, weight=1., alpha=0.25, gamma=2, eps=1e-12):
        self.weight = weight
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps

    def __call__(self, cls_pred, gt_labels):
        """
        Args:
            cls_pred (Tensor): Predicted classification logits, shape
                [num_query, num_class].
            gt_labels (Tensor): Label of `gt_bboxes`, shape (num_gt,).

        Returns:
            torch.Tensor: cls_cost value with weight
        """
        cls_pred = cls_pred.sigmoid()
        neg_cost = -(1 - cls_pred + self.eps).log() * (
            1 - self.alpha) * cls_pred.pow(self.gamma)
        pos_cost = -(cls_pred + self.eps).log() * self.alpha * (
            1 - cls_pred).pow(self.gamma)
        cls_cost = pos_cost[:, gt_labels] - neg_cost[:, gt_labels]
        return cls_cost * self.weight


@MATCH_COST.register_module()
class ClassificationCost:
    """ClsSoftmaxCost.

     Args:
         weight (int | float, optional): loss_weight

     Examples:
         >>> from mmdet.core.bbox.match_costs.match_cost import \
         ... ClassificationCost
         >>> import torch
         >>> self = ClassificationCost()
         >>> cls_pred = torch.rand(4, 3)
         >>> gt_labels = torch.tensor([0, 1, 2])
         >>> factor = torch.tensor([10, 8, 10, 8])
         >>> self(cls_pred, gt_labels)
         tensor([[-0.3430, -0.3525, -0.3045],
                [-0.3077, -0.2931, -0.3992],
                [-0.3664, -0.3455, -0.2881],
                [-0.3343, -0.2701, -0.3956]])
    """

    def __init__(self, weight=1.):
        self.weight = weight

    def __call__(self, cls_pred, gt_labels):
        """
        Args:
            cls_pred (Tensor): Predicted classification logits, shape
                [num_query, num_class].
            gt_labels (Tensor): Label of `gt_bboxes`, shape (num_gt,).

        Returns:
            torch.Tensor: cls_cost value with weight
        """
        # Following the official DETR repo, contrary to the loss that
        # NLL is used, we approximate it in 1 - cls_score[gt_label].
        # The 1 is a constant that doesn't change the matching,
        # so it can be omitted.
        cls_score = cls_pred.softmax(-1)
        cls_cost = -cls_score[:, gt_labels]
        return cls_cost * self.weight

def seesaw_func(cls_score: Tensor,
                   labels: Tensor,
                   cum_samples: Tensor,
                   num_classes: int,
                   p: float,
                   q: float,
                   eps: float) -> Tensor:
    assert cls_score.size(-1) == num_classes
    assert len(cum_samples) == num_classes

    onehot_labels = F.one_hot(labels, num_classes)
    seesaw_weights = cls_score.new_ones(onehot_labels.size())

    # mitigation factor
    if p > 0:
        sample_ratio_matrix = cum_samples[None, :].clamp(
            min=1) / cum_samples[:, None].clamp(min=1)
        index = (sample_ratio_matrix < 1.0).float()
        sample_weights = sample_ratio_matrix.pow(p) * index + (1 - index)
        mitigation_factor = sample_weights[labels.long(), :]
        seesaw_weights = seesaw_weights * mitigation_factor

    # compensation factor
    if q > 0:
        scores = F.softmax(cls_score.detach(), dim=1)
        self_scores = scores[
            torch.arange(0, len(scores)).to(scores.device).long(),
            labels.long()]
        score_matrix = scores / self_scores[:, None].clamp(min=eps)
        index = (score_matrix > 1.0).float()
        compensation_factor = score_matrix.pow(q) * index + (1 - index)
        seesaw_weights = seesaw_weights * compensation_factor

    cls_score = cls_score + (seesaw_weights.log() * (1 - onehot_labels))
    
    return cls_score

class BaseSeesawCost(nn.Module):
    """Base class for shared logic between a series of definitions of SeesawLossCost.

    Args:
        weight (int | float, optional): loss_weight
        p (float, optional): The parameter p for the seesaw loss.
        q (float, optional): The parameter q for the seesaw loss.
        num_classes (int, optional): The number of classes.
        eps (float, optional): The parameter eps for the seesaw loss.
        with_bg_score (bool, optional): Whether the background class is included in the score.
    """

    def __init__(self,
                 weight: float = 1.0,
                 p: float = 0.8,
                 q: float = 2.0,
                 num_classes: int = 18,
                 eps: float = 1e-2,
                 with_bg_score: bool = False) -> None:
        super().__init__()
        self.weight = weight
        self.p = p
        self.q = q
        self.num_classes = num_classes
        self.eps = eps
        self.with_bg_score = with_bg_score

        # cumulative samples for each category
        if not with_bg_score:
            self.register_buffer(
                'cum_samples',
                torch.zeros(self.num_classes, dtype=torch.float, device='cuda'))
        else:
            self.register_buffer(
                'cum_samples',
                torch.zeros(self.num_classes + 1, dtype=torch.float, device='cuda'))

    def _split_cls_score(self, cls_score: Tensor) -> Tuple[Tensor, Tensor]:
        """Split cls_score.

        Args:
            cls_score (Tensor): The prediction with shape (N, C + 1).

        Returns:
            Tuple[Tensor, Tensor]: The score for classes and objectness,
                respectively.
        """
        assert cls_score.size(-1) == self.num_classes + 1
        cls_score_classes = cls_score[..., :-1]
        cls_score_objectness = torch.stack((cls_score[..., :-1].max(dim=-1).values,
                                            cls_score[..., -1]), dim=-1)
        return cls_score_classes, cls_score_objectness

    def forward(self, 
                 cls_score: Tensor,
                 gt_labels: Tensor,
                 cost_function) -> Union[Tensor, Dict[str, Tensor]]:
        """
        Args:
            cls_pred (Tensor): Predicted classification logits, shape
                [num_query, num_class or num_classes + 1].
            gt_labels (Tensor): Label of `gt_bboxes`, shape (num_gt,).
            cost_function: Cost function to compute the classification loss.

        Returns:
            torch.Tensor: cls_cost value with weight.
        """
        if not self.with_bg_score:
            assert cls_score.size(-1) == self.num_classes
        else:
            assert cls_score.size(-1) == self.num_classes + 1
        if self.cum_samples.device != gt_labels.device:
            self.cum_samples = self.cum_samples.to(gt_labels.device)
        pos_inds = gt_labels < self.num_classes
        # Accumulate the samples for each category
        unique_labels = gt_labels.unique()
        for u_l in unique_labels:
            inds_ = gt_labels == u_l.item()
            self.cum_samples[u_l] += inds_.sum()

        if not self.with_bg_score:
            assert pos_inds.sum() == len(gt_labels)
            cls_score = seesaw_func(cls_score, gt_labels, self.cum_samples, self.num_classes, 
                                    self.p, self.q, self.eps)
            cls_classes_cost = cost_function(cls_score, gt_labels)
            return cls_classes_cost
            
        # 0 for pos, 1 for neg
        obj_labels = (gt_labels == self.num_classes).long()

        cls_score_classes, cls_score_objectness = self._split_cls_score(cls_score)
        if pos_inds.sum() > 0:
            cls_score_classes_ = seesaw_func(cls_score_classes[pos_inds], gt_labels[pos_inds],
                                            self.cum_samples[:self.num_classes], self.num_classes, self.p, 
                                            self.q, self.eps)
            cls_classes_cost = cost_function(cls_score_classes_, gt_labels[pos_inds])
        else:
            cls_classes_cost = self.weight * cls_score_classes[pos_inds].sum()
        cls_objectness_cost = -self.weight * cls_score_objectness.softmax(-1)[:, obj_labels]
        cls_cost = cls_classes_cost + cls_objectness_cost
        
        return cls_cost


@MATCH_COST.register_module()
class SeesawLossCost(BaseSeesawCost):
    """The naive SeesawLossCost.
    
    """

    def __init__(self,
                 weight: float = 1.0,
                 p: float = 0.8,
                 q: float = 2.0,
                 num_classes: int = 18,
                 eps: float = 1e-2,
                 with_bg_score: bool = False) -> None:
        super().__init__(weight, p, q, num_classes, eps, with_bg_score)

    def forward(self, cls_score: Tensor, gt_labels: Tensor) -> Tensor:
        cost_function = ClassificationCost(self.weight)
        return super().forward(cls_score, gt_labels, cost_function)


@MATCH_COST.register_module()
class SeesawFocalLossCost(BaseSeesawCost):
    """SeesawFocalLossCost.

    Args:
        weight (int | float, optional): loss_weight
        p (float, optional): The parameter p for the seesaw loss.
        q (float, optional): The parameter q for the seesaw loss.
        num_classes (int, optional): The number of classes.
        eps (float, optional): The parameter eps for the seesaw loss.
        gamma (float, optional): The gamma for calculating the modulating
            factor for focal loss. Defaults to 2.0.
        alpha (float, optional): A balanced form for Focal Loss.
            Defaults to 0.25.
        with_bg_score (bool, optional): Whether the background class is included in the score.

    """

    def __init__(self,
                 weight: float = 1.0,
                 p: float = 0.8,
                 q: float = 2.0,
                 num_classes: int = 18,
                 eps: float = 1e-2,
                 gamma: float = 2.0,
                 alpha: float = 0.25,
                 with_bg_score: bool = True) -> None:
        super().__init__(weight, p, q, num_classes, eps, with_bg_score)
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, cls_score: Tensor, gt_labels: Tensor) -> Tensor:
        cost_function = FocalLossCost(self.weight, self.alpha, self.gamma, self.eps)
        return super().forward(cls_score, gt_labels, cost_function)


@MATCH_COST.register_module()
class IoUCost:
    """IoUCost.

     Args:
         iou_mode (str, optional): iou mode such as 'iou' | 'giou'
         weight (int | float, optional): loss weight

     Examples:
         >>> from mmdet.core.bbox.match_costs.match_cost import IoUCost
         >>> import torch
         >>> self = IoUCost()
         >>> bboxes = torch.FloatTensor([[1,1, 2, 2], [2, 2, 3, 4]])
         >>> gt_bboxes = torch.FloatTensor([[0, 0, 2, 4], [1, 2, 3, 4]])
         >>> self(bboxes, gt_bboxes)
         tensor([[-0.1250,  0.1667],
                [ 0.1667, -0.5000]])
    """

    def __init__(self, iou_mode='giou', weight=1.):
        self.weight = weight
        self.iou_mode = iou_mode

    def __call__(self, bboxes, gt_bboxes):
        """
        Args:
            bboxes (Tensor): Predicted boxes with unnormalized coordinates
                (x1, y1, x2, y2). Shape [num_query, 4].
            gt_bboxes (Tensor): Ground truth boxes with unnormalized
                coordinates (x1, y1, x2, y2). Shape [num_gt, 4].

        Returns:
            torch.Tensor: iou_cost value with weight
        """
        # overlaps: [num_bboxes, num_gt]
        overlaps = bbox_overlaps(
            bboxes, gt_bboxes, mode=self.iou_mode, is_aligned=False)
        # The 1 is a constant that doesn't change the matching, so omitted.
        iou_cost = -overlaps
        return iou_cost * self.weight

@MATCH_COST.register_module()
class SoftmaxFocalLossCost:
    """FocalLossCost.

     Args:
         weight (int | float, optional): loss_weight
         gamma (int | float, optional): focal_loss gamma
         eps (float, optional): default 1e-12

     Examples:
         >>> from mmdet.core.bbox.match_costs.match_cost import SoftmaxFocalLossCost
         >>> import torch
         >>> self = SoftmaxFocalLossCost()
         >>> cls_pred = torch.rand(4, 3)
         >>> gt_labels = torch.tensor([0, 1, 2])
         >>> factor = torch.tensor([10, 8, 10, 8])
         >>> self(cls_pred, gt_labels)
         tensor([[-0.3236, -0.3364, -0.2699],
                [-0.3439, -0.3209, -0.4807],
                [-0.4099, -0.3795, -0.2929],
                [-0.1950, -0.1207, -0.2626]])
    """

    def __init__(self, weight=1., gamma=1.5, eps=1e-12):
        self.weight = weight
        self.gamma = gamma
        self.eps = eps

    def __call__(self, cls_pred, gt_labels):
        """
        Args:
            cls_pred (Tensor): Predicted classification logits, shape
                [num_query, num_class].
            gt_labels (Tensor): Label of `gt_bboxes`, shape (num_gt,).

        Returns:
            torch.Tensor: cls_cost value with weight
        """
        # import ipdb;ipdb.set_trace()
        # focal loss
        cls_score = cls_pred.softmax(-1)
        cls_cost = -cls_score[:, gt_labels]
        return cls_cost * self.weight



@MATCH_COST.register_module()
class SoftFocalLossCost:
    """FocalLossCost.

     Args:
         weight (int | float, optional): loss_weight
         alpha (int | float, optional): focal_loss alpha
         gamma (int | float, optional): focal_loss gamma
         eps (float, optional): default 1e-12

     Examples:
         >>> from mmdet.core.bbox.match_costs.match_cost import FocalLossCost
         >>> import torch
         >>> self = FocalLossCost()
         >>> cls_pred = torch.rand(4, 3)
         >>> gt_labels = torch.tensor([0, 1, 2])
         >>> factor = torch.tensor([10, 8, 10, 8])
         >>> self(cls_pred, gt_labels)
         tensor([[-0.3236, -0.3364, -0.2699],
                [-0.3439, -0.3209, -0.4807],
                [-0.4099, -0.3795, -0.2929],
                [-0.1950, -0.1207, -0.2626]])
    """

    def __init__(self, weight=1., alpha=0.25, gamma=2, eps=1e-12, soft_option=0):
        self.soft_option = soft_option
        self.weight = weight
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps

    def __call__(self, cls_pred, gt_labels, gt_scores=None):
        """
        Args:
            cls_pred (Tensor): Predicted classification logits, shape
                [num_query, num_class].
            gt_labels (Tensor): Label of `gt_bboxes`, shape (num_gt,).

        Returns:
            torch.Tensor: cls_cost value with weight
        """
        if gt_scores is None:
            cls_pred = cls_pred.sigmoid()
            neg_cost = -(1 - cls_pred + self.eps).log() * (
                1 - self.alpha) * cls_pred.pow(self.gamma)
            pos_cost = -(cls_pred + self.eps).log() * self.alpha * (
                1 - cls_pred).pow(self.gamma)
            cls_cost = pos_cost[:, gt_labels] - neg_cost[:, gt_labels]
            return cls_cost * self.weight
        else:
            prob = cls_pred.sigmoid()   # [N, num_class]
            # 将标量列表表示的label转换为one-hot向量，并去除最后一维的背景类
            one_hot_label = prob.new_zeros(len(gt_labels), len(prob[0])).scatter_(1, gt_labels.unsqueeze(1), 1)    # [num_gt, num_class]
            # 其中soft_label的一行要么只有一个label被激活，要么全为0
            soft_label = gt_scores.unsqueeze(-1) * one_hot_label   # [num_gt, num_class] with soft score as the classification label
            # import ipdb;ipdb.set_trace()
            # 将对应的预测值和target值做成多维的
            num_pred = prob.size(0)
            num_gt = soft_label.size(0)
            prob = prob[:, None, :].repeat(1, num_gt, 1)                    # [N, 1, num_class]
            soft_label = soft_label[None, :, :].repeat(num_pred, 1, 1)      # [1, num_gt, num_class]
            neg_cost = -(1 - prob + self.eps).log() * (1 - soft_label) * torch.pow(soft_label, self.gamma)
            pos_cost = -(prob + self.eps).log() * soft_label * torch.pow(torch.abs(soft_label - prob), self.gamma)
            if self.soft_option == 0:
                neg_cost = neg_cost.sum(dim=-1)
                pos_cost = pos_cost.sum(dim=-1)
                # 参考quality focal loss，计算soft label的focal loss
                cls_cost =  pos_cost - neg_cost
                # 用avg_factor来计算平均值
                return cls_cost * self.weight
            else:
                cls_cost = pos_cost - neg_cost
                return cls_cost[:, torch.arange(num_gt), gt_labels] * self.weight


@MATCH_COST.register_module()
class KLDivCost:
    """KL Divergence based cost calculation.
    """

    def __init__(self, weight=1., eps=1e-12):
        self.weight = weight
       
        self.eps = eps

    def __call__(self, cls_pred, gt_label, gt_scores=None):
        """
        Args:
            cls_pred (Tensor): Predicted classification logits, shape
                [num_query, num_class].
            gt_labels (Tensor): [num_gt]
            gt_scores (Tensor): [num_gt, num_class]
        Returns:
            torch.Tensor: cls_cost value with weight
        """
        prob = cls_pred.sigmoid()   # [N, num_class]

             
        
        # 将对应的预测值和target值做成多维的
        num_pred = prob.size(0)
        num_gt = gt_scores.size(0)
        # 取出对应gt_label的score
        tgt_scores = gt_scores[torch.arange(num_gt), gt_label]          # [num_gt]

        prob = prob[:, None, :].repeat(1, num_gt, 1)                    # [N, 1, num_class]
        gt_scores = gt_scores[None, :, :].repeat(num_pred, 1, 1)        # [1, num_gt, num_class]

        # 将每个dim=1的维度的每个元素视为是一个二维的分布，计算对应的KL Divergence  
        pos_cost = (gt_scores / (prob + self.eps) + self.eps).log() * gt_scores                  # [num_pred, num_gt, num_class]
        neg_cost = ((1 - gt_scores) / (1 - prob + self.eps) + self.eps).log() * (1 - gt_scores)  # [num_pred, num_gt, num_class]

        # 注意还要乘以对应的gt label处的score来缩放一下cost，避免出现不同的pseudo bbox由于实际是
        # 同一个位置的框而有相同的score vector
        cls_cost =  pos_cost.sum(dim=-1) + neg_cost.sum(dim=-1)                         # [num_pred, num_gt]
        cls_cost = cls_cost * tgt_scores[None, :].repeat(num_pred, 1)
        # 用avg_factor来计算平均值
        return cls_cost * self.weight
    

@MATCH_COST.register_module()
class RotatedIoUCost:
    """
    RotatedIoUCost.
    Args:
        iou_mode (str, optional): iou mode such as 'iou' | 'iof'
        weight (int | float, optional): loss weight
    Examples:
        >>> from mmdet.core.bbox.match_costs.match_cost import RotatedIoUCost
        >>> import torch
        >>> self = RotatedIoUCost()
        >>> bboxes = torch.FloatTensor([[1,1, 2, 2, 0], [2, 2, 3, 4, 0]])
        >>> gt_bboxes = torch.FloatTensor([[0, 0, 2, 4, 0], [1, 2, 3, 4, 0]])
        >>> self(bboxes, gt_bboxes)
        tensor([[-0.1250,  0.1667],
                [ 0.1667, -0.5000]])
    Returns:
        torch.Tensor: iou_cost value with weight
    """

    def __init__(self, iou_mode='iou', weight=1.):
        self.weight = weight
        self.iou_mode = iou_mode

    def __call__(self, bboxes, gt_bboxes):
        """
        Args:
            bbox_pred (Tensor): Predicted boxes with unnormalized coordinates
                (cx, cy, w, h, angle). Shape [num_query, 5].
            gt_bboxes (Tensor): Ground truth boxes with unnormalized coordinates
                (cx, cy, w, h, angle). Shape [num_gt, 5].

        Returns:
            torch.Tensor: iou_cost value with weight
        """
        # overlaps: [num_bboxes, num_gt]
        overlaps = obb_overlaps(bboxes, gt_bboxes, mode=self.iou_mode, is_aligned=False)
        # The 1 is a constant that doesn't change the matching, so omitted.
        iou_cost = -overlaps
        return iou_cost * self.weight
