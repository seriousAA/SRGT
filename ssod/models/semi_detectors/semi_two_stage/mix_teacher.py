import copy

import torch
import os
import random

from mmcv.runner import force_fp32, get_dist_info
from mmdet.core import multi_apply, arb2roi, bbox2type, get_bbox_dim, \
                        build_assigner, multiclass_arb_nms
from mmdet.models import BaseDetector, TwoStageDetector, DETECTORS, build_detector
from mmdet.models.builder import build_loss
from mmdet.models.losses import accuracy
from mmdet.ops import obb_overlaps, obb_nms

from ssod.utils.structure_utils import dict_split, weighted_loss
from ssod.utils import log_every_n
from ssod.models.utils import filter_invalid, process_visualization, visualize_images

from .multi_stream_detector import MultiStreamDetector


@DETECTORS.register_module()
class MixTeacher(MultiStreamDetector):
    def __init__(self, model: dict, train_cfg=None, test_cfg=None):
        super(MixTeacher, self).__init__(
            dict(teacher=build_detector(model), student=build_detector(model)),
            train_cfg=train_cfg,
            test_cfg=test_cfg,
        )
        self.freeze("teacher")
        if self.train_cfg is not None:
            self.unsup_weight = self.train_cfg.unsup_weight
            self.use_hbb_rpn = self.train_cfg.get("use_hbb_rpn", False)
            cls_loss_cfg = self.train_cfg.get("unsup_rcnn_cls", None)
            self.unsup_rcnn_cls = build_loss(cls_loss_cfg) if cls_loss_cfg else None
            initial_assigner_cfg = dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                match_low_quality=False,
                ignore_iof_thr=-1,
                iou_calculator=dict(type='OBBOverlaps'))
            self.initial_assigner = build_assigner(initial_assigner_cfg)
            self.distribute_sup_samples = self.train_cfg.get("distribute_sup_samples", None)
            self.total_labeled_batch = self.train_cfg.get("total_labeled_batch", None)
            self.distribute_unsup_samples = self.train_cfg.get("distribute_unsup_samples", None)
            self.total_unlabel_batch = self.train_cfg.get("total_unlabel_batch", None)
        
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

    def forward_train(self, img, img_metas, **kwargs):
        super().forward_train(img, img_metas, **kwargs)
        kwargs.update({"img": img})
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

    def forward_unsup_train(self, teacher_data, student_data, visualize=False):
        tea_img = teacher_data["img"]
        stu_img = student_data["img"]
        tea_img_metas = teacher_data["img_metas"]
        stu_img_metas = student_data["img_metas"]

        # sort the teacher and student input to avoid some bugs
        tnames = [meta["filename"] for meta in tea_img_metas]
        snames = [meta["filename"] for meta in stu_img_metas]
        tidx = [tnames.index(name) for name in snames]
        tea_img = tea_img[torch.Tensor(tidx).to(tea_img.device).long()]
        tea_img_metas = [tea_img_metas[idx] for idx in tidx]

        with torch.no_grad():
            tea_feat, tea_img_metas = self.teacher.extract_test_feat(tea_img, tea_img_metas)
            det_bboxes, det_labels, tea_proposals, tea_feats, _ = self.extract_teacher_info(
                tea_feat, tea_img_metas,
            )

        stu_hr_info = {'img': stu_img, 'img_metas': stu_img_metas}
        stu_lr_info = self.student.resize_info(stu_hr_info, scale=0.5)

        stu_hr_features = self.student.extract_feat(stu_hr_info['img'])
        stu_lr_features = self.student.extract_feat(stu_lr_info['img'])
        stu_fr_features = self.student.extract_fuse_feat(stu_hr_features, stu_lr_features)

        multi_res_info = {
            'hr': dict(x=stu_hr_features, **stu_hr_info),
            'lr': dict(x=stu_lr_features, **stu_lr_info),
            'fr': dict(x=stu_fr_features, **stu_hr_info)
        }
        if self.train_cfg.get('mine_pseudo_threshold', -1) > 0:
            multi_res_info['hr'].update(
                dict(ref_features=stu_hr_features, ref_img_metas=stu_hr_info['img_metas'],
                     tgt_features=stu_fr_features, tgt_img_metas=stu_hr_info['img_metas'])
            )

        ms_loss = {}
        for res_key, info in multi_res_info.items():
            loss = {}
            stu_feats, stu_img, stu_img_metas = info['x'], info['img'], info['img_metas']
            pseudo_bboxes, valid_masks = self.convert_bbox_space(tea_img_metas, stu_img_metas, 'obb', det_bboxes)
            pseudo_labels = [label[mask] for label, mask in zip(det_labels, valid_masks)]
            pseudo_proposals, _ = self.convert_bbox_space(tea_img_metas, stu_img_metas, 'hbb', tea_proposals)
            rpn_loss, stu_proposals = self.rpn_loss(stu_feats, pseudo_bboxes, stu_img_metas)
            loss.update(rpn_loss)

            if 'ref_features' not in info:
                tgt_cls_scores, tgt_decoded_bboxes = None, None
                ref_cls_scores, ref_decoded_bboxes = None, None
            else:
                ref_feats, ref_img_metas = info['ref_features'], info['ref_img_metas']
                tgt_feats, tgt_img_metas = info['tgt_features'], info['tgt_img_metas']
                tgt_proposals, _ = self.convert_bbox_space(
                    stu_img_metas, tgt_img_metas, 'hbb', stu_proposals, no_shape_filter=True)
                tgt_cls_scores, tgt_decoded_bboxes = self.extract_proposal_prediction(
                    tgt_feats, tgt_img_metas, tgt_proposals)
                tgt_decoded_bboxes, _ = self.convert_bbox_space(
                    tgt_img_metas, stu_img_metas, 'obb', tgt_decoded_bboxes, no_shape_filter=True)

                ref_proposals, _ = self.convert_bbox_space(
                    stu_img_metas, ref_img_metas, 'hbb', stu_proposals, no_shape_filter=True)
                ref_cls_scores, ref_decoded_bboxes = self.extract_proposal_prediction(
                    ref_feats, ref_img_metas, ref_proposals)
                ref_decoded_bboxes, _ = self.convert_bbox_space(
                    ref_img_metas, stu_img_metas, 'obb', ref_decoded_bboxes, no_shape_filter=True)

            if self.train_cfg.use_teacher_proposal:
                stu_proposals = pseudo_proposals

            loss.update(
                self.unsup_rcnn_cls_loss(
                    stu_feats,
                    stu_img_metas,
                    stu_proposals,
                    pseudo_bboxes,
                    pseudo_labels,
                    tea_img_metas,
                    tea_feats,
                    ref_decoded_bboxes,
                    ref_cls_scores,
                    tgt_decoded_bboxes,
                    tgt_cls_scores
                )
            )
            loss.update(
                self.unsup_rcnn_reg_loss(
                    stu_feats,
                    stu_img_metas,
                    stu_proposals,
                    pseudo_bboxes,
                    pseudo_labels,
                )
            )
            ms_loss.update({k + f'_{res_key}': v for k, v in loss.items()})
        ms_loss = self.student.reduce_losses(ms_loss, len(multi_res_info))
        return ms_loss

    def rpn_loss(self, stu_feats, pseudo_bboxes, img_metas, gt_bboxes_ignore=None, ):
        rpn_out = self.student.rpn_head(stu_feats)
        rpn_out = list(rpn_out)
        if self.student.with_rpn:
            gt_bboxes, _, _ = multi_apply(
                filter_invalid,
                [bbox[:, :5] for bbox in pseudo_bboxes],
                [None for _ in range(len(pseudo_bboxes))],
                [bbox[:, 5] for bbox in pseudo_bboxes],
                thr=self.train_cfg.rpn_pseudo_threshold,
                min_size=self.train_cfg.min_pseduo_box_size,
            )
            log_every_n(
                {"rpn_gt_num": sum([len(bbox) for bbox in gt_bboxes]) / len(gt_bboxes)}
            )
            if self.use_hbb_rpn:
                gt_bboxes_rpn = [bbox2type(bbox.float(), 'hbb') for bbox in gt_bboxes]
            else:
                gt_bboxes_rpn = [bbox.float() for bbox in gt_bboxes]
            loss_inputs = rpn_out + [gt_bboxes_rpn, img_metas]
            losses = self.student.rpn_head.loss(
                *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore
            )
            proposal_cfg = self.student.train_cfg.get(
                "rpn_proposal", self.student.test_cfg.rpn
            )
            proposal_list = self.student.rpn_head.get_bboxes(
                *rpn_out, img_metas=img_metas, cfg=proposal_cfg
            )
            return losses, proposal_list
        else:
            return {}, None

    def unsup_rcnn_cls_loss(
            self,
            feat,
            img_metas,
            proposal_list,
            pseudo_bboxes,
            pseudo_labels,
            teacher_img_metas,
            teacher_feat,
            ref_bboxes,
            ref_scores,
            tgt_bboxes,
            tgt_scores
    ):
        sampling_results, gt_bboxes, gt_labels = self.get_sampling_result_mine(
            img_metas,
            proposal_list,
            pseudo_bboxes,
            pseudo_labels,
            ref_bboxes,
            ref_scores,
            tgt_bboxes,
            tgt_scores,
            gt_bboxes_ignore=None,
        )
        selected_bboxes = [res.bboxes[:, :5] for res in sampling_results] # still compatiable for hbb proposals
        # rois = bbox2roi(selected_bboxes)
        rois = arb2roi(selected_bboxes, bbox_type=self.student.roi_head.bbox_head.start_bbox_type)
        bbox_results = self.student.roi_head._bbox_forward(feat, rois)
        bbox_targets = self.student.roi_head.bbox_head.get_targets(
            # 'gt_bboxes' should be 'hbb', but actually it won't be used in the training targets obtaining
            sampling_results, gt_bboxes, gt_labels, self.student.train_cfg.rcnn
        )
        # modify the negative proposal target
        aligned_proposals, _ = self.convert_bbox_space(
            img_metas, teacher_img_metas, 'hbb', selected_bboxes, no_shape_filter=True)
        with torch.no_grad():
            _, _scores = self.teacher.roi_head.simple_test_bboxes(
                teacher_feat,
                teacher_img_metas,
                aligned_proposals,
                None,
                rescale=False,
            )
            bg_score = torch.cat([_score[:, -1] for _score in _scores])
            assigned_label, _, _, _ = bbox_targets
            neg_inds = assigned_label == self.student.roi_head.bbox_head.num_classes
            bbox_targets[1][neg_inds] = bg_score[neg_inds].detach()
        if self.unsup_rcnn_cls:
            avg_factor = max(torch.sum(bbox_targets[1] > 0).float().item(), 1.)
            loss = {"loss_cls": self.unsup_rcnn_cls(
                            bbox_results["cls_score"],
                            bbox_targets[0],
                            bbox_targets[1],
                            avg_factor=avg_factor
                    )}
        else:
            loss = self.student.roi_head.bbox_head.loss(
                bbox_results["cls_score"],
                bbox_results["bbox_pred"],
                rois,
                *bbox_targets
        )
        
        return loss

    def unsup_rcnn_reg_loss(
        self,
        feat,
        img_metas,
        proposal_list,
        pseudo_bboxes,
        pseudo_labels,
        student_info=None,
        **kwargs,
    ):
        gt_bboxes, gt_labels, _ = multi_apply(
            filter_invalid,
            [bbox[:, :5] for bbox in pseudo_bboxes],
            pseudo_labels,
            [-bbox[:, 6:].mean(dim=-1) for bbox in pseudo_bboxes],
            thr=-self.train_cfg.reg_pseudo_threshold,
        )
        log_every_n(
            {"rcnn_reg_gt_num": sum([len(bbox) for bbox in gt_bboxes]) / len(gt_bboxes)}
        )
        loss_bbox = self.student.roi_head.forward_train(
            feat, img_metas, proposal_list, [bbox2type(gt_bbox, 'hbb') for gt_bbox in gt_bboxes]
            , gt_bboxes, gt_labels, **kwargs
        )["loss_bbox"]
        return {"loss_bbox": loss_bbox}

    def get_sampling_result_mine(
            self,
            img_metas,
            proposal_list,
            gt_obboxes,
            gt_labels,
            ref_bboxes,
            ref_scores,
            tgt_bboxes,
            tgt_scores,
            gt_bboxes_ignore=None,
    ):
        if ref_bboxes is None:
            return self.get_sampling_result_original(img_metas, proposal_list, gt_obboxes, gt_labels)

        num_imgs = len(img_metas)
        if gt_bboxes_ignore is None:
            gt_bboxes_ignore = [None for _ in range(num_imgs)]
        sampling_results = []
        for i in range(num_imgs):
            gt_bboxes_per_img = gt_obboxes[i][:, :5]
            gt_scores_per_img = gt_obboxes[i][:, 5]
            gt_labels_per_img = gt_labels[i]

            bboxes_ind = gt_scores_per_img > self.train_cfg.cls_pseudo_threshold
            gt_bboxes_per_img_high = gt_bboxes_per_img[bboxes_ind]
            gt_bboxes_per_img_high_ = gt_bboxes_per_img_high
            if self.use_hbb_rpn:
                gt_bboxes_per_img_high_ = bbox2type(gt_bboxes_per_img_high, 'hbb')
            gt_labels_per_img_high = gt_labels_per_img[bboxes_ind]

            # general label assignment with high confidence pseudo boxes
            assign_result = self.student.roi_head.bbox_assigner.assign(
                proposal_list[i], gt_bboxes_per_img_high_, gt_bboxes_ignore[i], gt_labels_per_img_high)

            # mine positive for all proposals assigned to negative
            neg_inds = assign_result.gt_inds == 0
            ref_bboxes_neg_per_img = ref_bboxes[i][neg_inds]
            tgt_bboxes_neg_per_img = tgt_bboxes[i][neg_inds]
            ref_scores_neg_per_img = ref_scores[i][neg_inds]
            tgt_scores_neg_per_img = tgt_scores[i][neg_inds]

            bboxes_ind = torch.bitwise_and(
                gt_scores_per_img < self.train_cfg.cls_pseudo_threshold,
                gt_scores_per_img > self.train_cfg.mine_pseudo_threshold
            )
            gt_bboxes_per_img_low = gt_bboxes_per_img[bboxes_ind]
            gt_labels_per_img_low = gt_labels_per_img[bboxes_ind]

            # assign reference bboxes with low confidence pseudo boxes
            assign_result_low = self.initial_assigner.assign(
                ref_bboxes_neg_per_img, gt_bboxes_per_img_low, None, gt_labels_per_img_low)

            # for all ref bboxes assigned to positive
            gt_inds = assign_result_low.gt_inds
            pos_inds = gt_inds > 0

            assigned_gt_inds = gt_inds - 1
            pos_assigned_gt_inds = assigned_gt_inds[pos_inds]
            pos_labels = gt_labels_per_img_low[pos_assigned_gt_inds]
            pos_tgt_scores_per_img = tgt_scores_neg_per_img[pos_inds]
            pos_ref_scores_per_img = ref_scores_neg_per_img[pos_inds]
            pos_tgt_bboxes_per_img = tgt_bboxes_neg_per_img[pos_inds]
            pos_ref_bboxes_per_img = ref_bboxes_neg_per_img[pos_inds]

            ref_ious = obb_overlaps(pos_ref_bboxes_per_img, gt_bboxes_per_img_low)
            tgt_ious = obb_overlaps(pos_tgt_bboxes_per_img, gt_bboxes_per_img_low)

            # refine assignment
            gt_inds_set = torch.unique(pos_assigned_gt_inds)
            mined_cnt = 0
            for gt_ind in gt_inds_set:
                pos_idx_per_gt = torch.nonzero(pos_assigned_gt_inds == gt_ind).reshape(-1)
                target_labels = pos_labels[pos_idx_per_gt]
                ref_scores_per_gt = pos_ref_scores_per_img[pos_idx_per_gt, target_labels]
                tgt_scores_per_gt = pos_tgt_scores_per_img[pos_idx_per_gt, target_labels]
                tgt_ious_per_gt = tgt_ious[pos_idx_per_gt, gt_ind]
                ref_ious_per_gt = ref_ious[pos_idx_per_gt, gt_ind]

                gt_bboxes_per_gt = gt_bboxes_per_img_low[gt_ind:gt_ind + 1]
                tgt_bboxes_pos_per_gt = pos_tgt_bboxes_per_img[pos_idx_per_gt]

                diff_score = (tgt_scores_per_gt - ref_scores_per_gt)
                bboxes_ind = diff_score > self.train_cfg.diff_score_threshold

                # traceback to the original proposals indices
                indices = torch.arange(len(ref_bboxes[i]), dtype=gt_inds.dtype, device=gt_inds.device)
                mined_ind = indices[neg_inds][pos_inds][pos_idx_per_gt][bboxes_ind]

                if len(mined_ind) == 0:
                    continue
                # add mined results to the original assign results
                assign_result.gt_inds[mined_ind] = assign_result.num_gts + 1
                assign_result.num_gts += 1
                assign_result.labels[mined_ind] = target_labels[0]
                assign_result.max_overlaps[mined_ind] = ref_ious_per_gt[bboxes_ind]
                gt_bboxes_per_img_high = torch.cat([gt_bboxes_per_img_high, gt_bboxes_per_gt])
                gt_labels_per_img_high = torch.cat([gt_labels_per_img_high, target_labels[0:1]])
                mined_cnt += 1

            log_every_n({"mined_cls_bboxes": mined_cnt})
            gt_bboxes = list(gt_obboxes)
            gt_labels = list(gt_labels)
            gt_bboxes[i] = gt_bboxes_per_img_high
            gt_labels[i] = gt_labels_per_img_high

            sampling_result = self.student.roi_head.bbox_sampler.sample(
                assign_result,
                proposal_list[i],
                bbox2type(gt_bboxes[i], 'hbb') if self.use_hbb_rpn else gt_bboxes[i],
                gt_labels[i]
            )

            if self.use_hbb_rpn:
                if gt_bboxes[i].numel() == 0:
                    sampling_result.pos_gt_bboxes = gt_bboxes[i].new_zeros(
                        (0, gt_bboxes[0].size(-1)))
                else:
                    sampling_result.pos_gt_bboxes = \
                            gt_bboxes[i][sampling_result.pos_assigned_gt_inds, :]            
            
            sampling_results.append(sampling_result)
        return sampling_results, gt_bboxes, gt_labels

    def get_sampling_result_original(
            self,
            img_metas,
            proposal_list,
            pseudo_bboxes,
            pseudo_labels,
            gt_bboxes_ignore=None,
    ):
        gt_obboxes, gt_labels, _ = multi_apply(
            filter_invalid,
            [bbox[:, :5] for bbox in pseudo_bboxes],
            pseudo_labels,
            [bbox[:, 5] for bbox in pseudo_bboxes],
            thr=self.train_cfg.cls_pseudo_threshold,
        )
        log_every_n(
            {"rcnn_cls_gt_num": sum([len(bbox) for bbox in gt_obboxes]) / len(gt_obboxes)}
        )
        num_imgs = len(img_metas)
        if gt_bboxes_ignore is None:
            gt_bboxes_ignore = [None for _ in range(num_imgs)]
        gt_bboxes = gt_obboxes
        if self.use_hbb_rpn:
            gt_bboxes = [bbox2type(obbox.float(), 'hbb') for obbox in gt_obboxes]

        sampling_results = []
        for i in range(num_imgs):
            assign_result = self.student.roi_head.bbox_assigner.assign(
                proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i], gt_labels[i]
            )
            sampling_result = self.student.roi_head.bbox_sampler.sample(
                assign_result,
                proposal_list[i],
                gt_bboxes[i],
                gt_labels[i],
            )
            if self.use_hbb_rpn:
                if gt_obboxes[i].numel() == 0:
                    sampling_result.pos_gt_bboxes = gt_obboxes[i].new_zeros(
                        (0, gt_obboxes[0].size(-1)))
                else:
                    sampling_result.pos_gt_bboxes = \
                            gt_obboxes[i][sampling_result.pos_assigned_gt_inds, :]

            sampling_results.append(sampling_result)
        return sampling_results, gt_obboxes, gt_labels

    @torch.no_grad()
    def assignment_refinement(self, gt_inds, pos_inds, pos_assigned_gt_inds,
                              ious, cls_score, areas, labels):
        # (PLA) refine assignment results according to teacher predictions
        # on each image
        refined_gt_inds = gt_inds.new_full((gt_inds.shape[0],), -1)
        refined_pos_gt_inds = gt_inds.new_full((pos_inds.shape[0],), -1)

        gt_inds_set = torch.unique(pos_assigned_gt_inds)
        for gt_ind in gt_inds_set:
            # for cluster with class k `k=labels[pos_idx_per_gt]`,
            pos_idx_per_gt = torch.nonzero(pos_assigned_gt_inds == gt_ind).reshape(-1)
            target_labels = labels[pos_idx_per_gt]  # should be same for all proposals in cluster
            target_scores = cls_score[pos_idx_per_gt, target_labels]  # scores of class k for all proposals in cluster
            target_areas = areas[pos_idx_per_gt]  # areas for all proposals in cluster
            target_IoUs = ious[pos_idx_per_gt, gt_ind]  # ious wrt cluster center for all proposals in cluster

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

    def extract_proposal_prediction(self, feat, img_metas, proposal_list):
        bbox_preds_list, cls_scores_list = self.teacher.roi_head.simple_test_bboxes(
            feat, img_metas, proposal_list, None, rescale=False
        )
        return cls_scores_list, bbox_preds_list

    def extract_teacher_info(self, feat, img_metas):
        proposal_cfg = self.teacher.train_cfg.get(
            "rpn_proposal_cfg", self.teacher.test_cfg.rpn
        )
        rpn_out = list(self.teacher.rpn_head(feat))
        proposal_list = self.teacher.rpn_head.get_bboxes(
            *rpn_out, img_metas=img_metas, cfg=proposal_cfg
        )
        proposals = copy.deepcopy(proposal_list)

        # split RCNN prediction in to forward and  NMS step
        # proposal_list, proposal_label_list = self.teacher.roi_head.simple_test_bboxes(
        #     feat, img_metas, proposal_list, self.teacher.test_cfg.rcnn, rescale=False
        # )
        cls_scores_list, bbox_preds_list = self.extract_proposal_prediction(
            feat, img_metas, proposal_list)

        det_bboxes_list, det_labels_list = [], []
        cfg = self.teacher.test_cfg.rcnn
        end_bbox_type = self.teacher.roi_head.bbox_head.end_bbox_type
        for bbox_preds, cls_scores in zip(bbox_preds_list, cls_scores_list):
            det_bboxes, det_labels = multiclass_arb_nms(bbox_preds, cls_scores,
                                                    cfg.score_thr, cfg.nms, cfg.max_per_img,
                                                    bbox_type=end_bbox_type)
            det_bboxes_list.append(det_bboxes)
            det_labels_list.append(det_labels)

        det_bboxes = [b.to(feat[0].device) for b in det_bboxes_list]
        det_bboxes = [
            b if b.shape[0] > 0 else b.new_zeros(0, 6) for b in det_bboxes
        ]
        det_labels = [b.to(feat[0].device) for b in det_labels_list]
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
        reg_unc = self.compute_uncertainty_with_aug(
            feat, img_metas, det_bboxes, det_labels
        )
        det_bboxes = [
            torch.cat([bbox, unc], dim=-1) for bbox, unc in zip(det_bboxes, reg_unc)
        ]
        
        return det_bboxes, det_labels, proposals, feat, cls_scores_list

    def compute_uncertainty_with_aug(
        self, feat, img_metas, proposal_list, proposal_label_list
    ):
        auged_proposal_list = self.aug_box(
            proposal_list, self.train_cfg.jitter_times, self.train_cfg.jitter_scale
        )
        # flatten
        auged_proposal_list = [
            auged.reshape(-1, auged.shape[-1]) if not self.use_hbb_rpn 
                else self.bbox2type_with_score(auged.reshape(-1, auged.shape[-1]).float(), 'hbb') 
                    for auged in auged_proposal_list
        ]
        with torch.no_grad():
            bboxes, _ = self.teacher.roi_head.simple_test_bboxes(
                feat,
                img_metas,
                auged_proposal_list,
                None,
                rescale=False,
            )
        reg_channel = max([bbox.shape[-1] for bbox in bboxes]) // 5
        bboxes = [
            bbox.reshape(self.train_cfg.jitter_times, -1, bbox.shape[-1])
            if bbox.numel() > 0
            else bbox.new_zeros(self.train_cfg.jitter_times, 0, 5 * reg_channel).float()
            for bbox in bboxes
        ]

        box_unc = [bbox.std(dim=0) for bbox in bboxes]
        bboxes = [bbox.mean(dim=0) for bbox in bboxes]
        # scores = [score.mean(dim=0) for score in scores]
        if reg_channel != 1:
            bboxes = [
                bbox.reshape(bbox.shape[0], reg_channel, 5)[
                    torch.arange(bbox.shape[0]), label
                ]
                for bbox, label in zip(bboxes, proposal_label_list)
            ]
            box_unc = [
                unc.reshape(unc.shape[0], reg_channel, 5)[
                    torch.arange(unc.shape[0]), label
                ]
                for unc, label in zip(box_unc, proposal_label_list)
            ]

        box_shape = [bbox[:, None, 2:4].clamp(min=1.0).expand(-1, 2, 2).reshape(-1, 4) for bbox in bboxes]
        # relative unc
        box_unc = [
            unc / torch.cat([wh, torch.full((wh.shape[0], 1), torch.pi).to(wh.device)], dim=-1)
            if wh.numel() > 0
            else unc
            for unc, wh in zip(box_unc, box_shape)
        ]
        return box_unc

    @staticmethod
    def aug_box(boxes, times=1, frac=0.06):
        def _aug_single(box):
            # random translate
            # TODO: random flip or something
            # box_scale = box[:, 2:4] - box[:, :2]
            # box_scale = (
            #     box_scale.clamp(min=1)[:, None, :].expand(-1, 2, 2).reshape(-1, 4)
            # )
            # aug_scale = box_scale * frac  # [n,4]

            aug_scale = box[:, 2:4].clamp(min=1) * frac  # [n,2]
            offset = (
                torch.randn(times, box.shape[0], 2, device=box.device)
                * aug_scale[None, ...]
            )
            new_box = box.clone()[None, ...].expand(times, box.shape[0], -1)
            return torch.cat(
                [new_box[:, :, :2].clone() + offset, new_box[:, :, 2:]], dim=-1
            )

        return [_aug_single(box) for box in boxes]
