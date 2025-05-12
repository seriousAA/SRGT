import cv2
import copy
import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import mmcv
import random

from PIL import Image
from ssod.utils import log_every_n
from mmdet.datasets.pipelines.obb.misc import visualize_with_obboxes, vis_args
import BboxToolkit as bt
from torchvision.transforms import ToPILImage
transform = ToPILImage()

from mmcv.runner import force_fp32, get_dist_info
from mmdet.core import multi_apply, OBBOverlaps, build_assigner, \
                        arb2roi, bbox2type, get_bbox_dim, BboxOverlaps2D
from mmdet.models import BaseDetector, TwoStageDetector, DETECTORS, build_detector
from mmdet.models.builder import build_loss
from mmdet.models.losses import accuracy
from mmdet.ops.nms_rotated import arb_batched_nms
from mmdet.datasets.pipelines import Compose

from ssod.utils.structure_utils import dict_split, weighted_loss
from mmdet.models.losses.utils import weight_reduce_loss
from BboxToolkit.geometry import bbox_overlaps, bbox_areas

from .multi_stream_detector import MultiStreamDetector
from ssod.models.utils import Transform2D, filter_invalid, resize_image, visualize_images

@DETECTORS.register_module()
class PseCo(MultiStreamDetector):
    """ PseCo on FR-CNN.
    """
    def __init__(self, model: dict, train_cfg=None, test_cfg=None):
        super(PseCo, self).__init__(
            dict(teacher=build_detector(model), student=build_detector(model)),
            train_cfg=train_cfg,
            test_cfg=test_cfg,
        )
        self.freeze("teacher")
        if self.train_cfg is not None:
            self.unsup_weight = self.train_cfg.unsup_weight

            # initialize assignment to build condidate bags
            self.PLA_iou_thres = self.train_cfg.get("PLA_iou_thres", 0.4)
            initial_assigner_cfg=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=self.PLA_iou_thres,
                neg_iou_thr=self.PLA_iou_thres,
                match_low_quality=False,
                ignore_iof_thr=-1,
                iou_calculator=dict(type='OBBOverlaps'))
            self.initial_assigner = build_assigner(initial_assigner_cfg)
            self.PLA_candidate_topk = self.train_cfg.PLA_candidate_topk

            self.use_teacher_proposal = self.train_cfg.use_teacher_proposal
            self.use_MSL = self.train_cfg.use_MSL
            self.use_hbb_rpn = self.train_cfg.use_hbb_rpn if hasattr(self.train_cfg, 'use_hbb_rpn') else False
            cls_loss_cfg = self.train_cfg.get("unsup_rcnn_cls", None)
            self.unsup_rcnn_cls = build_loss(cls_loss_cfg) if cls_loss_cfg else None
            self.distribute_sup_samples = self.train_cfg.get("distribute_sup_samples", None)
            self.total_labeled_batch = self.train_cfg.get("total_labeled_batch", None)
            self.distribute_unsup_samples = self.train_cfg.get("distribute_unsup_samples", None)
            self.total_unlabel_batch = self.train_cfg.get("total_unlabel_batch", None)


        if self.student.roi_head.bbox_head.loss_cls.use_sigmoid:
            self.use_sigmoid = True
        else:
            self.use_sigmoid = False

        self.num_classes = self.student.roi_head.bbox_head.num_classes
        self.rcnn_test_cfg = self.teacher.test_cfg.rcnn
        self.start_bbox_type = self.teacher.roi_head.bbox_head.start_bbox_type
        self.end_bbox_type = self.teacher.roi_head.bbox_head.end_bbox_type
        # cancel the gt_as_proposals while sampling proposals by pseudo bboxes
        self.student.roi_head.bbox_sampler.add_gt_as_proposals = False

    def sample_training_data(self, data_groups, num_samples, indices=None):
        """
        Randomly sample a specific number of training samples from data_groups
        
        Args:
            data_groups (dict): Dictionary containing the training data
            num_samples (int): Number of samples to select
            
        Returns:
            dict: A new dictionary with the same structure but containing only the sampled data
        """
        # Get the total number of samples in the original data
        total_samples = len(data_groups["gt_bboxes"])
        
        # Make sure num_samples is not larger than total samples
        assert num_samples <= total_samples
        
        if indices is None:
            # Generate random indices
            indices = random.sample(range(total_samples), num_samples)
        
        # Create a new dictionary with the sampled data
        sampled_data = {
            "gt_bboxes": [data_groups["gt_bboxes"][i] for i in indices],
            "gt_obboxes": [data_groups["gt_obboxes"][i] for i in indices],
            "gt_labels": [data_groups["gt_labels"][i] for i in indices],
            "img": data_groups["img"][indices],
            "img_metas": [data_groups["img_metas"][i] for i in indices]
        }
        
        return sampled_data, indices

    def forward_train(self, imgs, img_metas, **kwargs):
        super().forward_train(imgs, img_metas, **kwargs)

        kwargs.update({"img": imgs})
        kwargs.update({"img_metas": img_metas})
        kwargs.update({"tag": [meta["tag"] for meta in img_metas]})
        data_groups = dict_split(kwargs, "tag")
        for _, v in data_groups.items():
            v.pop("tag")

        loss = {}

        cur_iter = self.cur_iter
        if "unsup_start_iter" in os.environ:
            unsup_start = int(os.environ["unsup_start_iter"])

        #! Warnings: By splitting losses for supervised data and unsupervised data with different names,
        #! it means that at least one sample for each group should be provided on each gpu.
        #! In some situation, we can only put one image per gpu, we have to return the sum of loss
        #! and log the loss with logger instead. Or it will try to sync tensors don't exist.
        if self.distribute_sup_samples is not None and self.distribute_unsup_samples is not None:
            rank, world_size = get_dist_info()
            assert len(self.distribute_sup_samples) == len(self.distribute_unsup_samples) == world_size
            assert sum(self.distribute_sup_samples) == self.total_labeled_batch and \
                    sum(self.distribute_unsup_samples) == self.total_unlabel_batch
            cur_sup_sample = self.distribute_sup_samples[rank]
            cur_unsup_sample = self.distribute_unsup_samples[rank]
            num_device_total = world_size
            
            if "sup" in data_groups:
                data_groups_sup = data_groups["sup"]
                if cur_sup_sample < len(data_groups["sup"]["gt_bboxes"]):
                    data_groups_sup, _ = self.sample_training_data(data_groups["sup"], max(1, cur_sup_sample))
                gt_bboxes = data_groups_sup["gt_obboxes"]
                log_every_n(
                    {"sup_gt_num": sum([len(bbox) for bbox in gt_bboxes]) / len(gt_bboxes) if len(gt_bboxes) > 0 else 0}
                )
                if cur_sup_sample == 0:
                    with torch.no_grad():
                        sup_loss = self.student.forward_train(**data_groups_sup)
                else:
                    sup_loss = self.student.forward_train(**data_groups_sup)
                sup_loss = {"sup_" + k: v for k, v in sup_loss.items()}
                for k, v in sup_loss.items():
                    if isinstance(v, torch.Tensor):
                        sup_loss[k] = v * num_device_total * cur_sup_sample / self.total_labeled_batch
                    elif isinstance(v, list):
                        sup_loss[k] = [item * num_device_total * cur_sup_sample / self.total_labeled_batch for item in v]
                loss.update(**sup_loss)
            if "unsup_student" in data_groups and ("unsup_start_iter" not in os.environ or cur_iter >= unsup_start):
                data_groups_unsup_stu = data_groups["unsup_student"]
                data_groups_unsup_tea = data_groups["unsup_teacher"]
                if cur_unsup_sample < len(data_groups["unsup_student"]["gt_bboxes"]):
                    data_groups_unsup_stu, indices = self.sample_training_data(data_groups["unsup_student"], max(1, cur_unsup_sample))
                    data_groups_unsup_tea, _ = self.sample_training_data(
                        data_groups["unsup_teacher"], max(1, cur_unsup_sample), indices=indices)

                if cur_unsup_sample == 0:
                    with torch.no_grad():
                        unsup_loss = weighted_loss(
                            self.forward_unsup_train(
                                data_groups_unsup_tea, data_groups_unsup_stu
                            ),
                            weight=self.unsup_weight,
                        )
                else:
                    unsup_loss = weighted_loss(
                        self.forward_unsup_train(
                            data_groups_unsup_tea, data_groups_unsup_stu
                        ),
                        weight=self.unsup_weight,
                    )
                unsup_loss = {"unsup_" + k: v for k, v in unsup_loss.items()}
                for k, v in unsup_loss.items():
                    if isinstance(v, torch.Tensor):
                        unsup_loss[k] = v * num_device_total * cur_unsup_sample / self.total_unlabel_batch
                    elif isinstance(v, list):
                        unsup_loss[k] = [item * num_device_total * cur_unsup_sample / self.total_unlabel_batch for item in v]
                loss.update(**unsup_loss)
        else:
            if "sup" in data_groups:
                gt_bboxes = data_groups["sup"]["gt_obboxes"]
                log_every_n(
                    {"sup_gt_num": sum([len(bbox) for bbox in gt_bboxes]) / len(gt_bboxes)}
                )
                sup_loss = self.student.forward_train(**data_groups["sup"])
                sup_loss = {"sup_" + k: v for k, v in sup_loss.items()}
                loss.update(**sup_loss)
            if "unsup_student" in data_groups and ("unsup_start_iter" not in os.environ or cur_iter >= unsup_start):
                unsup_loss = weighted_loss(
                    self.forward_unsup_train(
                        data_groups["unsup_teacher"], data_groups["unsup_student"]
                    ),
                    weight=self.unsup_weight,
                )
                unsup_loss = {"unsup_" + k: v for k, v in unsup_loss.items()}
                loss.update(**unsup_loss)

        return loss

    def extract_feat(self, img, model, start_lvl=0):
        """Directly extract features from the backbone+neck."""
        assert start_lvl in [0, 1], \
            f"start level {start_lvl} is not supported."
        x = model.backbone(img)
        # global feature -- [p2, p3, p4, p5, p6, p7]
        if model.with_neck:
            x = model.neck(x)

        upsampled_feature_map = F.interpolate(x[0], scale_factor=2, mode='nearest')
        x = (model.neck.fpn_convs[0](upsampled_feature_map),) + x
        if start_lvl == 0:
            return x[:-1]
        elif start_lvl == 1:
            return x[1:]

    def forward_sup_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_obboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_obboxes_ignore=None,
                      proposals=None,
                      **kwargs):
        """
        forward training process for the labeled data.
        """
        losses = dict()
        # high resolution
        x = self.extract_feat(img, self.student, start_lvl=1)
        # RPN forward and loss
        if self.student.with_rpn:
            proposal_type = getattr(self.student.rpn_head, 'bbox_type', 'hbb')
            target_bboxes = gt_bboxes if proposal_type == 'hbb' else gt_obboxes
            target_bboxes_ignore = gt_bboxes_ignore if proposal_type == 'hbb' \
                else gt_obboxes_ignore
            proposal_cfg = self.student.train_cfg.get('rpn_proposal',
                                              self.student.test_cfg.rpn)
            rpn_losses, proposal_list = self.student.rpn_head.forward_train(
                x,
                img_metas,
                target_bboxes,
                gt_labels=None,
                gt_bboxes_ignore=target_bboxes_ignore,
                proposal_cfg=proposal_cfg)
            losses.update(rpn_losses)
        else:
            proposal_list = proposals

        # RCNN forward and loss
        roi_losses = self.student.roi_head.forward_train(x, img_metas, proposal_list,
                                                 gt_bboxes, gt_obboxes, gt_labels,
                                                 gt_bboxes_ignore, gt_obboxes_ignore,
                                                 **kwargs)
        losses.update(roi_losses)
        return losses

    def forward_unsup_train(self, teacher_data, student_data, visualize=False):
        teacher_img = teacher_data["img"]
        student_img = student_data["img"]

        img_metas_teacher = teacher_data["img_metas"]
        img_metas_student = student_data["img_metas"]

        tea_gt_bboxes, tea_gt_labels = teacher_data["gt_obboxes"], teacher_data["gt_labels"]
        if len(img_metas_student) > 1:
            tnames = [meta["filename"] for meta in img_metas_teacher]
            snames = [meta["filename"] for meta in img_metas_student]
            tidx = [tnames.index(name) for name in snames]
            teacher_img = teacher_img[torch.Tensor(tidx).to(teacher_img.device).long()]
            img_metas_teacher = [img_metas_teacher[idx] for idx in tidx]

        with torch.no_grad():
            det_bboxes, det_labels, tea_proposals_tuple = self.extract_teacher_info(
                    teacher_img, img_metas_teacher)

        tea_proposals, tea_feats = tea_proposals_tuple
        tea_proposals, _ = self.convert_bbox_space(img_metas_teacher,
                         img_metas_student, 
                         'hbb' if self.use_hbb_rpn else 'obb',
                         tea_proposals)
        tea_gt_bboxes, valid_masks = self.convert_bbox_space(img_metas_teacher,
                         img_metas_student, 'obb', tea_gt_bboxes)
        tea_gt_labels = [label[mask] for label, mask in zip(tea_gt_labels, valid_masks)]

        pseudo_bboxes, valid_masks_ = self.convert_bbox_space(img_metas_teacher,
                         img_metas_student, 'obb', det_bboxes)
        pseudo_labels = [label[mask] for label, mask in zip(det_labels, valid_masks_)]
        
        loss = {}
        # RPN stage
        feats = self.extract_feat(student_img, self.student, start_lvl=1)
        stu_rpn_outs, rpn_losses = self.unsup_rpn_loss(
                feats, pseudo_bboxes, pseudo_labels, img_metas_student)
        loss.update(rpn_losses)

        if self.use_MSL:
            # construct View 2 to learn feature-level scale invariance
            # downsampled images by 0.5 or upsample images by 2.0
            img_ds = resize_image(student_img, resize_ratio=0.5)
            feats_ds = self.extract_feat(img_ds, self.student, start_lvl=0)
            _, rpn_losses_ds = self.unsup_rpn_loss(feats_ds,
                                    pseudo_bboxes, pseudo_labels,
                                    img_metas_student)
            for key, value in rpn_losses_ds.items():
                loss[key + "_V2"] = value

        # RCNN stage
        """ obtain proposals """
        if self.use_teacher_proposal:
            proposal_list = copy.deepcopy(tea_proposals)

        else :
            proposal_cfg = self.student.train_cfg.get(
                "rpn_proposal", self.student.test_cfg.rpn
            )
            proposal_list = self.student.rpn_head.get_bboxes(
                *stu_rpn_outs, img_metas_student, cfg=proposal_cfg
            )

        """ obtain teacher predictions for all proposals """
        with torch.no_grad():
            aligned_tea_proposals, valid_masks = self.convert_bbox_space(img_metas_student,
                                img_metas_teacher, 
                                'hbb' if self.use_hbb_rpn else 'obb',
                                tea_proposals, no_shape_filter=True)
            rois_ = arb2roi(aligned_tea_proposals, bbox_type=self.start_bbox_type)
            tea_bbox_results = self.teacher.roi_head._bbox_forward(
                             tea_feats, rois_)
        proposal_list = [proposal[mask] for proposal, mask in zip(proposal_list, valid_masks)]
        teacher_infos = {
            "imgs": teacher_img,
            "cls_score": tea_bbox_results["cls_score"].sigmoid() if self.use_sigmoid \
                else tea_bbox_results["cls_score"][:, :self.num_classes].softmax(dim=-1),
            "bbox_pred": tea_bbox_results["bbox_pred"],
            "feats": tea_feats,
            "img_metas": img_metas_teacher,
            "proposal_list": aligned_tea_proposals}

        rcnn_losses = self.unsup_rcnn_loss(
                            feats,
                            feats_ds if self.use_MSL else None,
                            img_metas_student,
                            proposal_list,
                            pseudo_bboxes,
                            pseudo_labels,
                            GT_bboxes=tea_gt_bboxes,
                            GT_labels=tea_gt_labels,
                            teacher_infos=teacher_infos)

        loss.update(rcnn_losses)
        
        return loss

    def unsup_rpn_loss(self, stu_feats, pseudo_bboxes, pseudo_labels, img_metas):
        stu_rpn_outs = self.student.rpn_head(stu_feats)
        device = stu_rpn_outs[0][0].device
        # rpn loss
        gt_bboxes_rpn = []
        for bbox, label in zip(pseudo_bboxes, pseudo_labels):
            bbox, label, _ = filter_invalid(
                bbox[:, :5],
                label=label,
                score=bbox[
                    :, 5
                ],  # TODO: replace with foreground score, here is classification score,
                thr=self.train_cfg.rpn_pseudo_threshold,
                min_size=self.train_cfg.min_pseduo_box_size,
            )
            gt_bboxes_rpn.append(bbox)
        gt_num_rpn = sum(gt_bbox_rpn.size(0) for gt_bbox_rpn in gt_bboxes_rpn) / len(gt_bboxes_rpn)

        if self.use_hbb_rpn:
            gt_bboxes_rpn_ = [bbox2type(bbox.float(), 'hbb') for bbox in gt_bboxes_rpn]
        else:
            gt_bboxes_rpn_ = [bbox.float() for bbox in gt_bboxes_rpn]
        stu_rpn_loss_inputs = stu_rpn_outs + (gt_bboxes_rpn_, img_metas)
        rpn_losses = self.student.rpn_head.loss(*stu_rpn_loss_inputs)
        rpn_losses["rpn_pseudo_bboxes_num"] = torch.tensor(gt_num_rpn, dtype=torch.float).to(device)
        return stu_rpn_outs, rpn_losses

    def unsup_rcnn_loss(self,
                        feat,
                        feat_V2,
                        img_metas,
                        proposal_list,
                        pseudo_bboxes,
                        pseudo_labels,
                        GT_bboxes=None,
                        GT_labels=None,
                        teacher_infos=None):
        device = feat[0].device

        gt_bboxes, gt_labels, _, idx = multi_apply(
            filter_invalid,
            [bbox[:, :5] for bbox in pseudo_bboxes],
            pseudo_labels,
            [bbox[:, 5] for bbox in pseudo_bboxes],
            thr=self.train_cfg.cls_pseudo_threshold,
            return_inds=True)
        gt_num = sum(gt_bbox.size(0) for gt_bbox in gt_bboxes) / len(gt_bboxes)

        sampling_results = self.prediction_guided_label_assign(
                    img_metas,
                    proposal_list,
                    gt_bboxes,
                    gt_labels,
                    teacher_infos=teacher_infos)

        selected_bboxes = [res.bboxes[:, :5] for res in sampling_results]
        pos_inds_list = [res.pos_inds for res in sampling_results]
        neg_inds_list = [res.neg_inds for res in sampling_results]
        pos_gt_bboxes_list = [res.pos_gt_bboxes for res in sampling_results]
        pos_assigned_gt_inds_list = [res.pos_assigned_gt_inds for res in sampling_results]

        bbox_targets = self.student.roi_head.bbox_head.get_targets(
            sampling_results, gt_bboxes, gt_labels, self.student.train_cfg.rcnn
        )
        labels = bbox_targets[0]

        rois = arb2roi(selected_bboxes, bbox_type=self.start_bbox_type)
        bbox_results = self.student.roi_head._bbox_forward(feat, rois)

        bbox_weights = self.compute_PCV(
                bbox_results["bbox_pred"],
                labels,
                selected_bboxes,
                pos_gt_bboxes_list,
                pos_assigned_gt_inds_list)
        bbox_weights_ = bbox_weights.pow(2.0)
        pos_inds = (labels >= 0) & (labels < self.num_classes)
        if pos_inds.any():
            reg_scale_factor = bbox_weights.sum() / (bbox_weights_.sum() + 1e-6)
        else:
            reg_scale_factor = 0.0

        if self.unsup_rcnn_cls:
            self.student.roi_head.bbox_head.loss_cls = self.unsup_rcnn_cls
        # Focal loss or Cross entropy loss
        loss = self.student.roi_head.bbox_head.loss(
            bbox_results["cls_score"],
            bbox_results["bbox_pred"],
            rois,
            *(bbox_targets[:3]),
            bbox_weights_,
            reduction_override="none",
        )
        
        loss["loss_cls"] = loss["loss_cls"].sum() / max(bbox_targets[1].sum(), 1.0)
        loss["loss_bbox"] = reg_scale_factor * loss["loss_bbox"].sum() / max(
            bbox_targets[1].size()[0], 1.0) 
        
        if feat_V2 is not None:
            bbox_results_V2 = self.student.roi_head._bbox_forward(feat_V2, rois)
            loss_V2 = self.student.roi_head.bbox_head.loss(
                bbox_results_V2["cls_score"],
                bbox_results_V2["bbox_pred"],
                rois,
                *(bbox_targets[:3]),
                bbox_weights_,
                reduction_override="none",
            )
            
            loss["loss_cls_V2"] = loss_V2["loss_cls"].sum() / max(bbox_targets[1].sum(), 1.0)
            loss["loss_bbox_V2"] = reg_scale_factor * loss_V2["loss_bbox"].sum() / max(
                bbox_targets[1].size()[0], 1.0) 
            if "acc" in loss_V2:
                loss["acc_V2"] = loss_V2["acc"]
            else:
                loss["acc_V2"] = accuracy(bbox_results_V2["cls_score"], labels)

        # print scores of positive proposals (analysis only)
        tea_cls_score = teacher_infos["cls_score"]
        num_proposal = [proposal.shape[0] for proposal in proposal_list]
        tea_cls_score_list = tea_cls_score.split(num_proposal, dim=0)   # tensor to list
        tea_pos_score = []
        for score, pos in zip(tea_cls_score_list, pos_inds_list):
            assert pos.numel() == 0 or pos.max() < score.shape[0]
            tea_pos_score.append(score[pos])
        tea_pos_score = torch.cat(tea_pos_score, dim=0)

        with torch.no_grad():
            if pos_inds.any():
                max_score = tea_pos_score[torch.arange(tea_pos_score.shape[0]), labels[pos_inds]].float()
                pos_score_mean = max_score.mean()
                pos_score_min = max_score.min()

            else:
                max_score = tea_cls_score.sum().float() * 0
                pos_score_mean = tea_cls_score.sum().float() * 0
                pos_score_min = tea_cls_score.sum().float() * 0

        loss["tea_pos_score_mean"] = pos_score_mean
        loss["tea_pos_score_min"] = pos_score_min
        loss['cls_score_thr'] = torch.tensor(self.train_cfg.cls_pseudo_threshold,
                                             dtype=torch.float,
                                             device=labels.device)
        loss["pos_num"] = pos_inds.sum().float() / len(pos_inds_list)
        loss["neg_num"] = torch.tensor(sum(neg_inds.size(0) for neg_inds in neg_inds_list) / len(neg_inds_list), dtype=torch.float).to(device)
        loss["pseudo_bboxes_num"] = torch.tensor(gt_num, dtype=torch.float).to(device)

        return loss


    def extract_teacher_info(self, img, img_metas, proposals=None, **kwargs):
        feat = self.extract_feat(img, self.teacher, start_lvl=1)
        
        if proposals is None:
            proposal_cfg = self.teacher.train_cfg.get(
                "rpn_proposal", self.teacher.test_cfg.rpn
            )
            rpn_out = list(self.teacher.rpn_head(feat))
            proposal_list = self.teacher.rpn_head.get_bboxes(
                *rpn_out, img_metas=img_metas, cfg=proposal_cfg
            )
        else:
            proposal_list = proposals

        det_bboxes, det_labels = self.teacher.roi_head.simple_test_bboxes(
            feat, img_metas, proposal_list, self.teacher.test_cfg.rcnn, rescale=False
        )

        det_bboxes = [b.to(feat[0].device) for b in det_bboxes]
        det_bboxes = [
            b if b.shape[0] > 0 else b.new_zeros(0, 6) for b in det_bboxes
        ]
        det_labels = [b.to(feat[0].device) for b in det_labels]
        # filter invalid box roughly
        if isinstance(self.train_cfg.pseudo_label_initial_score_thr, float):
            thr = self.train_cfg.pseudo_label_initial_score_thr
        else:
            # TODO: use dynamic threshold
            raise NotImplementedError("Dynamic Threshold is not implemented yet.")

        det_bboxes, det_labels, _ = list(
            zip(
                *[
                    filter_invalid(
                        proposal,
                        proposal_label,
                        proposal[:, -1],
                        thr=thr,
                        min_size=self.train_cfg.min_pseduo_box_size,
                    )
                    for proposal, proposal_label in zip(
                        det_bboxes, det_labels
                    )
                ]
            )
        )

        return det_bboxes, det_labels, \
            (proposal_list, feat)

    @torch.no_grad()
    def compute_PCV(self, 
                      bbox_preds, 
                      labels, 
                      proposal_list, 
                      pos_gt_bboxes_list, 
                      pos_assigned_gt_inds_list):
        """ Compute regression weights for each proposal according 
            to Positive-proposal Consistency Voting (PCV). 
        
        Args:
            bbox_pred (Tensors): bbox preds for proposals.
            labels (Tensors): assigned class label for each proposals. 
                0-79 indicate fg, 80 indicates bg.
            propsal_list tuple[Tensor]: proposals for each image.
            pos_gt_bboxes_list, pos_assigned_gt_inds_list tuple[Tensor]: label assignent results 
        
        Returns:
            bbox_weights (Tensors): Regression weights for proposals.
        """

        nums = [_.shape[0] for _ in proposal_list]
        labels = labels.split(nums, dim=0)
        bbox_preds = bbox_preds.split(nums, dim=0)
    
        bbox_weights_list = []

        for bbox_pred, label, proposals, pos_gt_bboxes, pos_assigned_gt_inds in zip(
                    bbox_preds, labels, proposal_list, pos_gt_bboxes_list, pos_assigned_gt_inds_list):

            pos_inds = ((label >= 0) & 
                        (label < self.num_classes)).nonzero().reshape(-1)
            bbox_weights = proposals.new_zeros(bbox_pred.shape[0], 5)
            pos_proposals = proposals[pos_inds]
            if len(pos_inds):
                pos_bbox_weights = proposals.new_zeros(pos_inds.shape[0], 5)
                pos_bbox_pred = bbox_pred[pos_inds]
                decoded_bboxes = self.student.roi_head.bbox_head.bbox_coder.decode(
                        pos_proposals, pos_bbox_pred)
                
                gt_inds_set = torch.unique(pos_assigned_gt_inds)

                iou_calculator = OBBOverlaps()
                IoUs = iou_calculator(
                    decoded_bboxes,
                    pos_gt_bboxes, mode='iou',
                    is_aligned=True)
    
                for gt_ind in gt_inds_set:
                    idx_per_gt = (pos_assigned_gt_inds == gt_ind).nonzero().reshape(-1)
                    if idx_per_gt.shape[0] > 0:
                        pos_bbox_weights[idx_per_gt] = IoUs[idx_per_gt].mean()
                bbox_weights[pos_inds] = pos_bbox_weights
               
            bbox_weights_list.append(bbox_weights)
        bbox_weights = torch.cat(bbox_weights_list, 0)
        
        return bbox_weights

    @torch.no_grad()
    def prediction_guided_label_assign(
                self,
                img_metas,
                proposal_list,
                gt_bboxes,
                gt_labels,
                teacher_infos,
                gt_bboxes_ignore=None,
    ):
        num_imgs = len(img_metas)
        if gt_bboxes_ignore is None:
            gt_bboxes_ignore = [None for _ in range(num_imgs)]

        # get teacher predictions (including cls scores and bbox ious)       
        tea_proposal_list = teacher_infos["proposal_list"]
        tea_cls_score_concat = teacher_infos["cls_score"]
        tea_bbox_pred_concat = teacher_infos["bbox_pred"]
        num_per_img = [_.shape[0] for _ in tea_proposal_list]
        tea_cls_scores = tea_cls_score_concat.split(num_per_img, dim=0)
        tea_bbox_preds = tea_bbox_pred_concat.split(num_per_img, dim=0)

        decoded_bboxes_list = self.get_tea_dense_bboxes(tea_bbox_preds, tea_proposal_list,
                                                        teacher_infos['img_metas'], img_metas)

        sampling_results = []
        for i in range(num_imgs):
            assign_result = self.initial_assigner.assign( 
                decoded_bboxes_list[i], 
                gt_bboxes[i], 
                gt_bboxes_ignore[i], 
                gt_labels[i])
            
            gt_inds = assign_result.gt_inds
            pos_inds = torch.nonzero(gt_inds > 0, as_tuple=False).reshape(-1)

            assigned_gt_inds = gt_inds - 1
            pos_assigned_gt_inds = assigned_gt_inds[pos_inds]        
            pos_labels = gt_labels[i][pos_assigned_gt_inds]
            
            tea_pos_cls_score = tea_cls_scores[i][pos_inds]
           
            tea_pos_bboxes = decoded_bboxes_list[i][pos_inds]
            iou_calculator = OBBOverlaps()
            ious = iou_calculator(tea_pos_bboxes, gt_bboxes[i])
            
            if self.use_hbb_rpn:
                wh = proposal_list[i][:, 2:4] - proposal_list[i][:, :2]
            else:
                wh = proposal_list[i][:, 2:4]
            areas = wh.prod(dim=-1)
            pos_areas = areas[pos_inds]
            
            refined_gt_inds = self.assignment_refinement(gt_inds, 
                                       pos_inds, 
                                       pos_assigned_gt_inds, 
                                       ious, 
                                       tea_pos_cls_score, 
                                       pos_areas, 
                                       pos_labels)
    
            assign_result.gt_inds = refined_gt_inds + 1
            sampling_result = self.student.roi_head.bbox_sampler.sample(
                                assign_result,
                                proposal_list[i],
                                bbox2type(gt_bboxes[i], 'hbb') if self.use_hbb_rpn else gt_bboxes[i],
                                gt_labels[i])

            if self.use_hbb_rpn:
                if gt_bboxes[i].numel() == 0:
                    sampling_result.pos_gt_bboxes = gt_bboxes[i].new_zeros(
                        (0, gt_bboxes[0].size(-1)))
                else:
                    sampling_result.pos_gt_bboxes = \
                            gt_bboxes[i][sampling_result.pos_assigned_gt_inds, :]            
            
            sampling_results.append(sampling_result)
        return sampling_results

    @torch.no_grad()
    def assignment_refinement(self, gt_inds, pos_inds, pos_assigned_gt_inds, 
                             ious, cls_score, areas, labels):
        # (PLA) refine assignment results according to teacher predictions 
        # on each image 
        refined_gt_inds = gt_inds.new_full((gt_inds.shape[0], ), -1)
        refined_pos_gt_inds = gt_inds.new_full((pos_inds.shape[0],), -1)
        
        gt_inds_set = torch.unique(pos_assigned_gt_inds)
        for gt_ind in gt_inds_set:
            pos_idx_per_gt = torch.nonzero(pos_assigned_gt_inds == gt_ind).reshape(-1)
            target_labels = labels[pos_idx_per_gt]
            target_scores = cls_score[pos_idx_per_gt, target_labels]
            target_areas = areas[pos_idx_per_gt]
            target_IoUs = ious[pos_idx_per_gt, gt_ind]
            
            cost = (target_IoUs * target_scores).sqrt()
            _, sort_idx = torch.sort(cost, descending=True)
            
            candidate_topk = min(pos_idx_per_gt.shape[0], self.PLA_candidate_topk)   
            topk_ious, _ = torch.topk(target_IoUs, candidate_topk, dim=0)
            # calculate dynamic k for each gt
            dynamic_ks = torch.clamp(topk_ious.sum(0).int(), min=1)      
            sort_idx = sort_idx[:dynamic_ks]
            # filter some invalid (area == 0) proposals
            sort_idx = sort_idx[
                target_areas[sort_idx] > 0
            ]
            pos_idx_per_gt = pos_idx_per_gt[sort_idx]
            
            refined_pos_gt_inds[pos_idx_per_gt] = pos_assigned_gt_inds[pos_idx_per_gt]
        
        refined_gt_inds[pos_inds] = refined_pos_gt_inds
        return refined_gt_inds

    def forward_test(self, imgs, img_metas, **kwargs):

        return super(MultiStreamDetector, self).forward_test(imgs, img_metas, **kwargs)

    def aug_test(self, imgs, img_metas, **kwargs):
        pass
    
    def simple_test(self, img, img_metas, proposals=None, rescale=False, **kwargs):
        """Test without augmentation."""
        
        model = self.model(**kwargs)
        assert model.with_bbox, 'Bbox head must be implemented.'
        
        x = self.extract_feat(img, model, start_lvl=1)

        if proposals is None:
            proposal_list = model.rpn_head.simple_test_rpn(x, img_metas)
        else:
            proposal_list = proposals

        return model.roi_head.simple_test(
            x, proposal_list, img_metas, rescale=rescale)

    def get_tea_dense_bboxes(self, tea_bbox_pred_list, tea_proposal_list, img_metas_teacher, img_metas_student):

        decoded_bboxes_list = []
        for bbox_preds, proposals in zip(tea_bbox_pred_list, tea_proposal_list):
            if bbox_preds.numel() > 0:
                decode_bboxes = self.student.roi_head.bbox_head.bbox_coder.decode(
                    proposals[:, :5], bbox_preds)
                decoded_bboxes_list.append(decode_bboxes)

        decoded_bboxes_list, _ = self.convert_bbox_space(
            img_metas_teacher,
            img_metas_student,
            'obb',
            decoded_bboxes_list,
            no_shape_filter=True)

        return decoded_bboxes_list
