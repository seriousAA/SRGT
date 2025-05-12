# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from functools import partial
from .cross_entropy_loss import cross_entropy
from .utils import weight_reduce_loss
from ..builder import LOSSES
from .focal_loss import sigmoid_focal_loss


def seesaw_ce_loss(cls_score: Tensor,
                   labels: Tensor,
                   label_weights: Tensor,
                   cum_samples: Tensor,
                   num_classes: int,
                   p: float,
                   q: float,
                   eps: float,
                   reduction: str = 'mean',
                   avg_factor: Optional[int] = None) -> Tensor:
    """Calculate the Seesaw CrossEntropy loss.

    Args:
        cls_score (Tensor): The prediction with shape (N, C),
             C is the number of classes.
        labels (Tensor): The learning label of the prediction.
        label_weights (Tensor): Sample-wise loss weight.
        cum_samples (Tensor): Cumulative samples for each category.
        num_classes (int): The number of classes.
        p (float): The ``p`` in the mitigation factor.
        q (float): The ``q`` in the compenstation factor.
        eps (float): The minimal value of divisor to smooth
             the computation of compensation factor
        reduction (str, optional): The method used to reduce the loss.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.

    Returns:
        Tensor: The calculated loss
    """
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

    loss = F.cross_entropy(cls_score, labels, weight=None, reduction='none')

    if label_weights is not None:
        label_weights = label_weights.float()
    loss = weight_reduce_loss(
        loss, weight=label_weights, reduction=reduction, avg_factor=avg_factor)
    return loss


def seesaw_focal_loss(cls_score: Tensor,
                   labels: Tensor,
                   label_weights: Tensor,
                   cum_samples: Tensor,
                   num_classes: int,
                   p: float,
                   q: float,
                   eps: float,
                   gamma: float = 2.0,
                   alpha: float = 0.25,
                   reduction: str = 'mean',
                   avg_factor: Optional[int] = None) -> Tensor:
    """Calculate the Seesaw CrossEntropy loss.

    Args:
        cls_score (Tensor): The prediction with shape (N, C),
             C is the number of classes.
        labels (Tensor): The learning label of the prediction.
        label_weights (Tensor): Sample-wise loss weight.
        cum_samples (Tensor): Cumulative samples for each category.
        num_classes (int): The number of classes.
        p (float): The ``p`` in the mitigation factor.
        q (float): The ``q`` in the compenstation factor.
        eps (float): The minimal value of divisor to smooth
             the computation of compensation factor
        gamma (float, optional): The gamma for calculating the modulating
            factor for focal loss. Defaults to 2.0.
        alpha (float, optional): A balanced form for Focal Loss.
            Defaults to 0.25.
        reduction (str, optional): The method used to reduce the loss.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.

    Returns:
        Tensor: The calculated loss
    """
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
    
    if label_weights is not None:
        label_weights = label_weights.float()

    loss = sigmoid_focal_loss(cls_score, labels, label_weights, 
                                gamma, alpha, reduction, avg_factor)
    return loss


@LOSSES.register_module()
class SeesawLoss(nn.Module):
    """
    Seesaw Loss for Long-Tailed Instance Segmentation (CVPR 2021)
    arXiv: https://arxiv.org/abs/2008.10032

    Args:
        use_sigmoid (bool, optional): Whether the prediction uses sigmoid
             of softmax. Only False is supported.
        p (float, optional): The ``p`` in the mitigation factor.
             Defaults to 0.8.
        q (float, optional): The ``q`` in the compenstation factor.
             Defaults to 2.0.
        num_classes (int, optional): The number of classes.
             Default to 1203 for LVIS v1 dataset.
        eps (float, optional): The minimal value of divisor to smooth
             the computation of compensation factor
        reduction (str, optional): The method that reduces the loss to a
             scalar. Options are "none", "mean" and "sum".
        loss_weight (float, optional): The weight of the loss. Defaults to 1.0
        return_dict (bool, optional): Whether return the losses as a dict.
             Default to True.
    """

    def __init__(self,
                 use_sigmoid: bool = False,
                 no_bg_score: bool = False,
                 p: float = 0.8,
                 q: float = 2.0,
                 num_classes: int = 18,
                 eps: float = 1e-2,
                 reduction: str = 'mean',
                 loss_weight: float = 1.0) -> None:
        super().__init__()
        assert not use_sigmoid
        self.use_sigmoid = False
        self.no_bg_score = no_bg_score
        self.p = p
        self.q = q
        self.num_classes = num_classes
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight

        # 0 for pos, 1 for neg
        self.cls_criterion = seesaw_ce_loss

        # cumulative samples for each category
        if self.no_bg_score:
            self.register_buffer(
                'cum_samples',
                torch.zeros(self.num_classes, dtype=torch.float))
        else:
            self.register_buffer(
                'cum_samples',
                torch.zeros(self.num_classes + 1, dtype=torch.float))

    def _split_cls_score(self, cls_score: Tensor) -> Tuple[Tensor, Tensor]:
        """split cls_score.

        Args:
            cls_score (Tensor): The prediction with shape (N, C + 1).

        Returns:
            Tuple[Tensor, Tensor]: The score for classes and objectness,
                respectively
        """
        # split cls_score to cls_score_classes and cls_score_objectness
        assert cls_score.size(-1) == self.num_classes + 1
        cls_score_classes = cls_score[..., :-1]
        cls_score_objectness = torch.stack((cls_score[..., :-1].max(dim=-1).values,
                                            cls_score[..., -1]), dim=-1)
        return cls_score_classes, cls_score_objectness

    def forward(
        self,
        cls_score: Tensor,
        labels: Tensor,
        label_weights: Optional[Tensor] = None,
        avg_factor: Optional[int] = None,
        reduction_override: Optional[str] = None
    ) -> Union[Tensor, Dict[str, Tensor]]:
        """Forward function.

        Args:
            cls_score (Tensor): The prediction with shape (N, C + 1).
            labels (Tensor): The learning label of the prediction.
            label_weights (Tensor, optional): Sample-wise loss weight.
            avg_factor (int, optional): Average factor that is used to average
                 the loss. Defaults to None.
            reduction (str, optional): The method used to reduce the loss.
                 Options are "none", "mean" and "sum".

        Returns:
            Tensor | Dict [str, Tensor]:
                 if return_dict == False: The calculated loss |
                 if return_dict == True: The dict of calculated losses
                 for objectness and classes, respectively.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if self.no_bg_score:
            assert cls_score.size(-1) == self.num_classes
        else:
            assert cls_score.size(-1) == self.num_classes + 1
        pos_inds = labels < self.num_classes
        # accumulate the samples for each category
        unique_labels = labels.unique()
        for u_l in unique_labels:
            inds_ = labels == u_l.item()
            self.cum_samples[u_l] += inds_.sum()

        if label_weights is not None:
            label_weights = label_weights.float()
        else:
            label_weights = labels.new_ones(labels.size(), dtype=torch.float)
            
        if self.no_bg_score:
            assert pos_inds.sum() == len(labels)
            loss_cls_classes = self.loss_weight * self.cls_criterion(
                cls_score[pos_inds], labels[pos_inds],
                label_weights[pos_inds], self.cum_samples[:self.num_classes],
                self.num_classes, self.p, self.q, self.eps, reduction=reduction,
                avg_factor=avg_factor)
            return loss_cls_classes
            
        # 0 for pos, 1 for neg
        obj_labels = (labels == self.num_classes).long()


        cls_score_classes, cls_score_objectness = self._split_cls_score(cls_score)

        # calculate loss_cls_classes (only need pos samples)
        if pos_inds.sum() > 0:
            loss_cls_classes = self.loss_weight * self.cls_criterion(
                cls_score_classes[pos_inds], labels[pos_inds],
                label_weights[pos_inds], self.cum_samples[:self.num_classes],
                self.num_classes, self.p, self.q, self.eps, reduction=reduction,
                avg_factor=avg_factor)
        else:
            loss_cls_classes = cls_score_classes[pos_inds].sum()

        # calculate loss_cls_objectness
        loss_cls_objectness = self.loss_weight * cross_entropy(
            cls_score_objectness, obj_labels, label_weights, reduction,
            avg_factor)

        loss_cls = loss_cls_classes + loss_cls_objectness
        
        return loss_cls

@LOSSES.register_module()
class SeesawFocalLoss(SeesawLoss):

    def __init__(self,
                 use_sigmoid: bool = False,
                 no_bg_score: bool = False,
                 p: float = 0.8,
                 q: float = 2.0,
                 num_classes: int = 18,
                 eps: float = 1e-2,
                 gamma: float = 2.0,
                 alpha: float = 0.25,
                 reduction: str = 'mean',
                 loss_weight: float = 1.0) -> None:
        super().__init__(use_sigmoid, no_bg_score, p, q, num_classes, eps, reduction, loss_weight)
        
        self.gamma = gamma
        self.alpha = alpha

        # 0 for pos, 1 for neg
        self.cls_criterion = partial(seesaw_focal_loss, gamma=self.gamma, alpha=self.alpha)

