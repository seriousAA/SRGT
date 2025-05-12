# Copyright (c) OpenMMLab. All rights reserved.
import torch
import numpy as np
import torch.nn.functional as F

from mmdet.core.bbox.builder import BBOX_ASSIGNERS
from mmdet.core.bbox.iou_calculators import bbox_overlaps
from mmdet.core.bbox.match_costs import build_match_cost
from mmdet.core.bbox.transforms import bbox_cxcywh_to_xyxy
from mmdet.core.bbox.assigners.assign_result import AssignResult
from mmdet.core.bbox.assigners.base_assigner import BaseAssigner
from mmdet.ops import obb_overlaps

from .o2m_assign_result import O2MAssignResult

@BBOX_ASSIGNERS.register_module()
class O2MAssigner(BaseAssigner):
    """Computes one-to-many matching between predictions and ground truth.

    This class computes an assignment between the targets and the predictions
    based on the costs. Supports both HBB and OBB annotations.

    Args:
        candidate_topk (int, optional): Number of top candidates for alignment. Default 13.
        debug (bool, optional): Flag for debugging. Default False.
    """

    def __init__(self, candidate_topk=13, debug=False):
        self.candidate_topk = candidate_topk
        self.debug = debug

    def assign(self,
               bbox_pred,
               cls_pred,
               gt_bboxes,
               gt_labels,
               img_meta,
               gt_bboxes_ignore=None,
               alpha=1,
               beta=6,
               teacher_assign=False,
               multiple_pos=False,
               bbox_type='hbb'):
        """Computes one-to-many matching based on the weighted costs.

        Args:
            bbox_pred (Tensor): Predicted boxes with normalized coordinates.
                For HBB: (cx, cy, w, h). For OBB: (cx, cy, w, h, a). Shape [num_query, 4 or 5].
            cls_pred (Tensor): Predicted classification logits. Shape [num_query, num_class].
            gt_bboxes (Tensor): Ground truth boxes with unnormalized coordinates.
                For HBB: (cx, cy, w, h). For OBB: (cx, cy, w, h, a). Shape [num_gt, 4 or 5].
            gt_labels (Tensor): Label of `gt_bboxes`, shape (num_gt,).
            img_meta (dict): Meta information for current image.
            gt_bboxes_ignore (Tensor, optional): Ground truth bboxes that are ignored. Default None.
            alpha (int | float, optional): Weight for classification score. Default 1.
            beta (int | float, optional): Weight for IoU overlap. Default 6.
            teacher_assign (bool, optional): Flag for teacher-student assignment. Default False.
            multiple_pos (bool, optional): Flag for allowing multiple positive matches. Default False.
            bbox_type (str, optional): Type of bounding boxes, either "hbb" or "obb". Default "hbb".

        Returns:
            :obj:`AssignResult`: The assigned result.
        """
        INF = 1e8
        assert gt_bboxes_ignore is None, \
            'Only case when gt_bboxes_ignore is None is supported.'

        num_gts, num_bboxes = gt_bboxes.size(0), bbox_pred.size(0)
        gt_labels = gt_labels.long()

        # Assign -1 by default
        assigned_gt_inds = bbox_pred.new_full((num_bboxes,), -1, dtype=torch.long)
        assigned_labels = bbox_pred.new_full((num_bboxes,), -1, dtype=torch.long)
        assign_metrics = bbox_pred.new_zeros((num_bboxes,))

        if num_gts == 0 or num_bboxes == 0:
            # No ground truth or boxes, return empty assignment
            max_overlaps = bbox_pred.new_zeros((num_bboxes,))
            if num_gts == 0:
                # No ground truth, assign all to background
                assigned_gt_inds[:] = 0
            return O2MAssignResult(
                num_gts, assigned_gt_inds, max_overlaps, assign_metrics, labels=assigned_labels)

        # Normalization factor based on bbox type
        img_h, img_w, _ = img_meta['img_shape']
        if bbox_type == 'obb':
            factor = gt_bboxes.new_tensor([img_w, img_h, img_w, img_h, np.pi]).unsqueeze(0)
        else:
            factor = gt_bboxes.new_tensor([img_w, img_h, img_w, img_h]).unsqueeze(0)

        # Convert predicted boxes based on bbox type
        if bbox_type == 'obb':
            temp_delta = bbox_pred.new_tensor([0., 0., 0., 0., 0.5]).unsqueeze(0)
            pred_bboxes = (bbox_pred - temp_delta) * factor
        else:
            pred_bboxes = bbox_cxcywh_to_xyxy(bbox_pred) * factor

        # Compute alignment metrics
        scores = cls_pred
        if bbox_type == 'obb':
            overlaps = obb_overlaps(pred_bboxes, gt_bboxes).detach()
        else:
            overlaps = bbox_overlaps(pred_bboxes, gt_bboxes).detach()

        bbox_scores = scores[:, gt_labels].detach()
        alignment_metrics = bbox_scores ** alpha * overlaps ** beta # [num_bbox, num_gt]

        # the top-k aligned predicted candidate boxes as the potential positive samples
        if teacher_assign and not multiple_pos:
            # option-1: only the top-aligned candidate box is kept as the positive samples
            _, candidate_idxs = alignment_metrics.topk(                 # [top-k, num_gt]
            1, dim=0, largest=True)
        else:  
            _, candidate_idxs = alignment_metrics.topk(                 # [top-k, num_gt]
                self.candidate_topk, dim=0, largest=True)
        candidate_metrics = alignment_metrics[candidate_idxs, torch.arange(num_gts)]

        # Handle multiple positives if needed
        if teacher_assign and multiple_pos:
            # option-2: dynamic estimate the positive samples for contrastive
            is_pos = torch.zeros_like(candidate_metrics)
            topk_ious, _ = torch.topk(overlaps, self.candidate_topk, dim=0) # [top-k, num_gt]
            dynamic_ks = torch.clamp(topk_ious.sum(0).int(), min=1)
            for gt_idx in range(num_gts):
                _, pos_idx = torch.topk(candidate_metrics[:, gt_idx], k=dynamic_ks[gt_idx].item(), largest=True)
                is_pos[:, gt_idx][pos_idx] = 1
            is_pos = is_pos.bool()
        else:
            is_pos = candidate_metrics > 0

        # Update indices
        for gt_idx in range(num_gts):
            candidate_idxs[:, gt_idx] += gt_idx * num_bboxes
        candidate_idxs = candidate_idxs.view(-1)

        # deal with a single candidate assigned to multiple gt_bboxes
        overlaps_inf = torch.full_like(overlaps,                    # [num_bbox x num_gt,]
                                       -INF).t().contiguous().view(-1)
        index = candidate_idxs.view(-1)[is_pos.view(-1)]            # [num_gt x top-k]
        overlaps_inf[index] = overlaps.t().contiguous().view(-1)[index]
        overlaps_inf = overlaps_inf.view(num_gts, -1).t()            # [num_gt, num_bbox] -> [num_bbox, num_gt]


        max_overlaps, argmax_overlaps = overlaps_inf.max(dim=1)

        # Assign backgrounds and foregrounds
        assigned_gt_inds[:] = 0
        assigned_gt_inds[max_overlaps != -INF] = argmax_overlaps[max_overlaps != -INF] + 1
        assign_metrics[max_overlaps != -INF] = alignment_metrics[max_overlaps != -INF, 
                                                                 argmax_overlaps[max_overlaps != -INF]]

        if gt_labels is not None:
            pos_inds = torch.nonzero(assigned_gt_inds > 0, as_tuple=False).squeeze()
            if pos_inds.numel() > 0:
                assigned_labels[pos_inds] = gt_labels[assigned_gt_inds[pos_inds] - 1]
        else:
            assigned_labels = None

        return O2MAssignResult(
            num_gts, assigned_gt_inds, max_overlaps, assign_metrics, labels=assigned_labels)
