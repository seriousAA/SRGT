import copy

import torch
import os
import math
from mmdet.core import multi_apply, arb2roi, bbox2type, get_bbox_dim, \
                        build_assigner, multiclass_arb_nms
from mmdet.models import BaseDetector, TwoStageDetector, DETECTORS, build_detector
from mmdet.models.builder import build_loss
from mmdet.models.losses import accuracy
from mmdet.ops import obb_overlaps, obb_nms

from ssod.utils.structure_utils import dict_split, weighted_loss
from ssod.utils import log_every_n
from ssod.models.utils import filter_invalid, process_visualization, \
                                compute_precision_recall_class_wise, \
                                collect_unique_labels_with_weights, \
                                calculate_average_metric_for_labels

from .multi_stream_detector import MultiStreamDetector


@DETECTORS.register_module()
class MixSRGT(MultiStreamDetector):
    def __init__(self, model: dict, train_cfg=None, test_cfg=None):
        super(MixSRGT, self).__init__(
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
            
            self.num_classes = self.student.roi_head.bbox_head.num_classes
            
            # Register the buffer
            self.register_buffer('gsd_mean_per_cls', torch.zeros((self.num_classes,), dtype=torch.float32))
            self.register_buffer('gsd_M2_per_cls', torch.zeros((self.num_classes,), dtype=torch.float32))
            self.register_buffer('gsd_W_per_cls', torch.zeros((self.num_classes,), dtype=torch.float32))
            self.register_buffer('gsd_mean_per_cls_unlabeled', torch.zeros((self.num_classes,), dtype=torch.float32))
            self.register_buffer('gsd_M2_per_cls_unlabeled', torch.zeros((self.num_classes,), dtype=torch.float32))
            self.register_buffer('gsd_W_per_cls_unlabeled', torch.zeros((self.num_classes,), dtype=torch.float32))
            self.register_buffer('gsd_mean_per_cls_unlabeled_gt', torch.zeros((self.num_classes,), dtype=torch.float32))
            self.register_buffer('gsd_M2_per_cls_unlabeled_gt', torch.zeros((self.num_classes,), dtype=torch.float32))
            self.register_buffer('gsd_W_per_cls_unlabeled_gt', torch.zeros((self.num_classes,), dtype=torch.float32))
            self.alpha = self.train_cfg.get("alpha", 0.01)
            self.beta = 1.0 - self.alpha
            self.base = self.train_cfg.get("base", 2)
            self.lmbda = self.train_cfg.get("lambda", 0.)
            self.kld_gamma = self.train_cfg.get("kld_gamma", None)
            self.rescale_pos_weight = self.train_cfg.get("rescale_pos_weight", False)
            self.no_unlabeled_gt = self.train_cfg.get("no_unlabeled_gt", False)

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
            unsup_warmup = int(os.environ["unsup_warmup_iter"])

        #! Warnings: By splitting losses for supervised data and unsupervised data with different names,
        #! it means that at least one sample for each group should be provided on each gpu.
        #! In some situation, we can only put one image per gpu, we have to return the sum of loss
        #! and log the loss with logger instead. Or it will try to sync tensors don't exist.
        if "sup" in data_groups:
            gt_bboxes = data_groups["sup"]["gt_obboxes"]
            log_every_n(
                {"sup_gt_num": sum([len(bbox) for bbox in gt_bboxes]) / len(gt_bboxes)}
            )
            sup_loss = self.student.forward_train(**data_groups["sup"])
            sup_loss = {"sup_" + k: v for k, v in sup_loss.items()}
            loss.update(**sup_loss)

        if "unsup_student" in data_groups and ("unsup_start_iter" not in os.environ or cur_iter >= unsup_start):
            unsup_loss, teacher_info = self.forward_unsup_train(
                data_groups["unsup_teacher"], data_groups["unsup_student"]
            )
            kld = self.gaussian_kld_divergence(self.gsd_mean_per_cls, (self.gsd_M2_per_cls / self.gsd_W_per_cls),
                                                          self.gsd_mean_per_cls_unlabeled, 
                                                          (self.gsd_M2_per_cls_unlabeled / self.gsd_W_per_cls_unlabeled),
                                                          normalize=False)
            normalized_kld = torch.full_like(kld, -1)
            normalized_kld[kld>=0] = torch.log1p(kld[kld>=0])
            kld_image = calculate_average_metric_for_labels(torch.cat(teacher_info['det_labels']), kld)
            normalized_kld_image = calculate_average_metric_for_labels(torch.cat(teacher_info['det_labels']), normalized_kld)
            if self.kld_gamma is not None and \
                ("unsup_start_iter" not in os.environ or cur_iter >= unsup_start+unsup_warmup) and \
                    kld_image >= 0:
                kld_weight = 1 / (1 + self.kld_gamma * normalized_kld_image)
                unsup_loss = weighted_loss(
                    unsup_loss,
                    weight=kld_weight,
                )
            else:
                unsup_loss = weighted_loss(
                    unsup_loss,
                    weight=self.unsup_weight,
                )
            unsup_loss = {"unsup_" + k: v for k, v in unsup_loss.items()}
            loss.update(**unsup_loss)
            dota2_0_names = ('large-vehicle', 'swimming-pool', 'helicopter', 'bridge',
                'plane', 'ship', 'soccer-ball-field', 'basketball-court',
                'ground-track-field', 'small-vehicle', 'baseball-diamond',
                'tennis-court', 'roundabout', 'storage-tank', 'harbor',
                'container-crane', 'helipad')
            # Log each class's statistics
            for cls_idx in range(self.num_classes):
                cls_name = dota2_0_names[cls_idx]
                loss.update({
                    f'gsd_{cls_name}_mean': self.gsd_mean_per_cls[cls_idx],
                    f'gsd_{cls_name}_M2': self.gsd_M2_per_cls[cls_idx],
                    f'gsd_{cls_name}_W': self.gsd_W_per_cls[cls_idx],
                    f'gsd_{cls_name}_var': (self.gsd_M2_per_cls / self.gsd_W_per_cls)[cls_idx] \
                        if self.gsd_W_per_cls[cls_idx] > 1e-6 
                        else torch.tensor(0.0, device=self.gsd_W_per_cls.device),
                })
                loss.update({
                    f'gsd_{cls_name}_mean_unlabeled': self.gsd_mean_per_cls_unlabeled[cls_idx],
                    f'gsd_{cls_name}_M2_unlabeled': self.gsd_M2_per_cls_unlabeled[cls_idx],
                    f'gsd_{cls_name}_W_unlabeled': self.gsd_W_per_cls_unlabeled[cls_idx],
                    f'gsd_{cls_name}_var_unlabeled': (self.gsd_M2_per_cls_unlabeled / self.gsd_W_per_cls_unlabeled)[cls_idx] \
                        if self.gsd_W_per_cls_unlabeled[cls_idx] > 1e-6 
                        else torch.tensor(0.0, device=self.gsd_W_per_cls_unlabeled.device),
                })
                if not self.no_unlabeled_gt:
                    loss.update({
                        f'gsd_{cls_name}_mean_unlabeled_gt': self.gsd_mean_per_cls_unlabeled_gt[cls_idx],
                        f'gsd_{cls_name}_M2_unlabeled_gt': self.gsd_M2_per_cls_unlabeled_gt[cls_idx],
                        f'gsd_{cls_name}_W_unlabeled_gt': self.gsd_W_per_cls_unlabeled_gt[cls_idx],
                        f'gsd_{cls_name}_var_unlabeled_gt': (self.gsd_M2_per_cls_unlabeled_gt / self.gsd_W_per_cls_unlabeled_gt)[cls_idx] \
                            if self.gsd_W_per_cls_unlabeled_gt[cls_idx] > 1e-6 
                            else torch.tensor(0.0, device=self.gsd_W_per_cls_unlabeled_gt.device),
                    })

        if "unsup_student" in data_groups and ("unsup_start_iter" not in os.environ or cur_iter >= unsup_start):
            if not self.no_unlabeled_gt:
                kld_gt = self.gaussian_kld_divergence(self.gsd_mean_per_cls, (self.gsd_M2_per_cls / self.gsd_W_per_cls),
                                                            self.gsd_mean_per_cls_unlabeled_gt, 
                                                            (self.gsd_M2_per_cls_unlabeled_gt / self.gsd_W_per_cls_unlabeled_gt),
                                                            normalize=False)
                normalized_kld_gt = torch.full_like(kld_gt, -1)
                normalized_kld_gt[kld_gt>=0] = torch.log1p(kld_gt[kld_gt>=0])
                kld_gt_image = calculate_average_metric_for_labels(torch.cat(teacher_info['gt_labels']), kld_gt)
                normalized_kld_gt_image = calculate_average_metric_for_labels(torch.cat(teacher_info['gt_labels']), normalized_kld_gt)
            
            self.update_unlabeled_gsd_stats(data_groups["unsup_teacher"], teacher_info)
            
            for cls_idx in range(self.num_classes):
                cls_name = dota2_0_names[cls_idx]
                if kld[cls_idx] >= 0:
                    loss.update({
                        f'gsd_kld_{cls_name}': kld[cls_idx],
                        f'gsd_normalized_kld_{cls_name}': normalized_kld[cls_idx],
                    })
                if (not self.no_unlabeled_gt) and kld_gt[cls_idx] >= 0:
                    loss.update({
                        f'gsd_kld_{cls_name}_gt': kld_gt[cls_idx],
                        f'gsd_normalized_kld_{cls_name}_gt': normalized_kld_gt[cls_idx],
                    })
            if kld_image >= 0:
                loss.update({
                    f'gsd_kld_image_level': kld_image,
                    f'gsd_normalized_kld_image_level': normalized_kld_image,
                })
            if (not self.no_unlabeled_gt) and kld_gt_image >= 0:
                loss.update({
                    f'gsd_kld_gt_image_level': kld_gt_image,
                    f'gsd_normalized_kld_gt_image_level': normalized_kld_gt_image,
                })
            if not self.no_unlabeled_gt:
                pseudo_label_quality = compute_precision_recall_class_wise(
                    [teacher_info['gt_obboxes'][0]], [teacher_info['gt_labels'][0]],
                    [teacher_info['det_bboxes'][0]], [teacher_info['det_labels'][0]],
                    iou_threshold=0.5
                )
                pseudo_label_quality_high = compute_precision_recall_class_wise(
                    [teacher_info['gt_obboxes'][0]], [teacher_info['gt_labels'][0]],
                    [teacher_info['det_bboxes'][0]], [teacher_info['det_labels'][0]],
                    iou_threshold=0.75
                )
            pseudo_label_weights = collect_unique_labels_with_weights(
                teacher_info['det_labels'][0], teacher_info["pseudo_label_weights"][0]
            )
            pseudo_label_weights_unclamped = collect_unique_labels_with_weights(
                teacher_info['det_labels'][0], teacher_info["pseudo_label_weights_unclamped"][0]
            )
            if not self.no_unlabeled_gt:
                pseudo_label_weights_gt = collect_unique_labels_with_weights(
                    teacher_info['gt_labels'][0], teacher_info["pseudo_label_weights_gt"][0]
                )
                pseudo_label_weights_gt_unclamped = collect_unique_labels_with_weights(
                    teacher_info['gt_labels'][0], teacher_info["pseudo_label_weights_gt_unclamped"][0]
                )
            for cls_idx in range(self.num_classes):
                cls_name = dota2_0_names[cls_idx]
                if not self.no_unlabeled_gt:
                    if cls_idx in pseudo_label_quality:
                        if 'precision' in pseudo_label_quality[cls_idx]:
                            precision = pseudo_label_quality[cls_idx]['precision']
                            loss.update({
                                f'pseudo_precision_{cls_name}': precision,
                            })
                        if 'recall' in pseudo_label_quality[cls_idx]:
                            recall = pseudo_label_quality[cls_idx]['recall']
                            loss.update({
                                f'pseudo_recall_{cls_name}': recall,
                            })
                    if cls_idx in pseudo_label_quality_high:
                        if 'precision' in pseudo_label_quality_high[cls_idx]:
                            precision_high = pseudo_label_quality_high[cls_idx]['precision']
                            loss.update({
                                f'pseudo_precision_{cls_name}_high': precision_high,
                            })
                        if 'recall' in pseudo_label_quality_high[cls_idx]:
                            recall_high = pseudo_label_quality_high[cls_idx]['recall']
                            loss.update({
                                f'pseudo_recall_{cls_name}_high': recall_high,
                            })
                    if cls_idx in pseudo_label_weights_gt:
                        loss.update({
                            f'pseudo_label_weight_gt_{cls_name}': pseudo_label_weights_gt[cls_idx],
                            f'pseudo_label_weight_gt_unclamped_{cls_name}': pseudo_label_weights_gt_unclamped[cls_idx],
                        })
                if cls_idx in pseudo_label_weights:
                    loss.update({
                        f'pseudo_label_weight_{cls_name}': pseudo_label_weights[cls_idx],
                        f'pseudo_label_weight_unclamped_{cls_name}': pseudo_label_weights_unclamped[cls_idx],
                    })
            if (not self.no_unlabeled_gt) and 'overall' in pseudo_label_quality:
                if 'precision' in pseudo_label_quality['overall']:
                    precision = pseudo_label_quality['overall']['precision']
                    loss.update({
                        'pseudo_precision_overall': precision,
                    })
                if 'recall' in pseudo_label_quality['overall']:
                    recall = pseudo_label_quality['overall']['recall']
                    loss.update({
                        'pseudo_recall_overall': recall,
                    })
                if 'precision' in pseudo_label_quality_high['overall']:
                    precision_high = pseudo_label_quality_high['overall']['precision']
                    loss.update({
                        'pseudo_precision_overall_high': precision_high,
                    })
                if 'recall' in pseudo_label_quality_high['overall']:
                    recall_high = pseudo_label_quality_high['overall']['recall']
                    loss.update({
                        'pseudo_recall_overall_high': recall_high,
                    })
        if "sup" in data_groups:
            self.update_labeled_gsd_stats(data_groups["sup"])
        return loss

    def update_labeled_gsd_stats(self, data_groups_sup):
        for idx, labels in enumerate(data_groups_sup["gt_labels"]):
            gsd_value_ori = data_groups_sup['img_metas'][idx]['img_info']['gsd']
            if 'scale_factor' in  data_groups_sup['img_metas'][idx]:
                gsd_value_ori /= data_groups_sup['img_metas'][idx]['scale_factor'].mean()
            elif 'img_shape' in data_groups_sup['img_metas'][idx]:
                scale_factor = (float(data_groups_sup['img_metas'][idx]['img_shape'][0])/data_groups_sup['img_metas'][idx]['img_info']['height']+
                                float(data_groups_sup['img_metas'][idx]['img_shape'][1])/data_groups_sup['img_metas'][idx]['img_info']['width'])/2.0
                gsd_value_ori /= scale_factor
            gsd_value = math.log((self.lmbda + gsd_value_ori), self.base)
            for label in labels.unique():
                label_count = (labels == label).sum().item()
                if self.gsd_W_per_cls[label] < 1e-6:
                    self.gsd_mean_per_cls[label] = gsd_value
                    self.gsd_M2_per_cls[label] = 0.0
                    self.gsd_W_per_cls[label] = float(label_count)
                else:
                    W_decayed = self.gsd_W_per_cls[label] * self.beta
                    M2_decayed = self.gsd_M2_per_cls[label] * self.beta
                    w = float(label_count)
                    W_new = W_decayed + w
                    delta = gsd_value - self.gsd_mean_per_cls[label]
                    self.gsd_mean_per_cls[label] += delta * (w / W_new)
                    self.gsd_M2_per_cls[label] = M2_decayed + delta * \
                        (gsd_value - self.gsd_mean_per_cls[label]) * w
                    self.gsd_W_per_cls[label] = W_new

    def update_unlabeled_gsd_stats(self, data_groups_unsup, teacher_info):
        for idx, labels in enumerate(data_groups_unsup["gt_labels"]):
            gsd_value_ori = data_groups_unsup['img_metas'][idx]['img_info']['gsd']
            if 'scale_factor' in  data_groups_unsup['img_metas'][idx]:
                gsd_value_ori /= data_groups_unsup['img_metas'][idx]['scale_factor'].mean()
            elif 'img_shape' in data_groups_unsup['img_metas'][idx]:
                scale_factor = (float(data_groups_unsup['img_metas'][idx]['img_shape'][0])/data_groups_unsup['img_metas'][idx]['img_info']['height']+
                                float(data_groups_unsup['img_metas'][idx]['img_shape'][1])/data_groups_unsup['img_metas'][idx]['img_info']['width'])/2.0
                gsd_value_ori /= scale_factor
            gsd_value = math.log((self.lmbda + gsd_value_ori), self.base)
            det_labels = teacher_info["det_labels"][idx]
            for label in det_labels.unique():
                label_count = (det_labels == label).sum().item()
                if self.gsd_W_per_cls_unlabeled[label] < 1e-6:
                    self.gsd_mean_per_cls_unlabeled[label] = gsd_value
                    self.gsd_M2_per_cls_unlabeled[label] = 0.0
                    self.gsd_W_per_cls_unlabeled[label] = float(label_count)
                else:
                    W_decayed = self.gsd_W_per_cls_unlabeled[label] * self.beta
                    M2_decayed = self.gsd_M2_per_cls_unlabeled[label] * self.beta
                    w = float(label_count)
                    W_new = W_decayed + w
                    delta = gsd_value - self.gsd_mean_per_cls_unlabeled[label]
                    self.gsd_mean_per_cls_unlabeled[label] += delta * (w / W_new)
                    self.gsd_M2_per_cls_unlabeled[label] = M2_decayed + delta * \
                        (gsd_value - self.gsd_mean_per_cls_unlabeled[label]) * w
                    self.gsd_W_per_cls_unlabeled[label] = W_new

            if not self.no_unlabeled_gt:
                for label in labels.unique():
                    label_count = (labels == label).sum().item()
                    if self.gsd_W_per_cls_unlabeled_gt[label] < 1e-6:
                        self.gsd_mean_per_cls_unlabeled_gt[label] = gsd_value
                        self.gsd_M2_per_cls_unlabeled_gt[label] = 0.0
                        self.gsd_W_per_cls_unlabeled_gt[label] = float(label_count)
                    else:
                        W_decayed = self.gsd_W_per_cls_unlabeled_gt[label] * self.beta
                        M2_decayed = self.gsd_M2_per_cls_unlabeled_gt[label] * self.beta
                        w = float(label_count)
                        W_new = W_decayed + w
                        delta = gsd_value - self.gsd_mean_per_cls_unlabeled_gt[label]
                        self.gsd_mean_per_cls_unlabeled_gt[label] += delta * (w / W_new)
                        self.gsd_M2_per_cls_unlabeled_gt[label] = M2_decayed + delta * \
                            (gsd_value - self.gsd_mean_per_cls_unlabeled_gt[label]) * w
                        self.gsd_W_per_cls_unlabeled_gt[label] = W_new
        
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
            teacher_inputs = {}
            teacher_inputs['feat'], teacher_inputs['img_metas'] = self.teacher.extract_test_feat(tea_img, tea_img_metas)
            if "gt_obboxes" in teacher_data and teacher_data["gt_obboxes"] is not None:
                teacher_inputs["gt_obboxes"] = [teacher_data["gt_obboxes"][idx] for idx in tidx]
                teacher_inputs["gt_labels"] = [teacher_data["gt_labels"][idx] for idx in tidx]
            det_bboxes, det_labels, tea_proposals, tea_feats, _, teacher_info = self.extract_teacher_info(
                **teacher_inputs
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
        return ms_loss, teacher_info

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
        
        labels = bbox_targets[0]
        bbox_targets_list = list(bbox_targets)
        with torch.no_grad():
            reweights = self.reweight_label_weights(labels, bbox_targets[1], img_metas)
        bbox_targets_list[1] = reweights
        bbox_targets = tuple(bbox_targets_list)
        if self.unsup_rcnn_cls:
            avg_factor = max(torch.sum(bbox_targets[1] > 0).float().item(), 1.)
            if not self.rescale_pos_weight:
                loss = {"loss_cls": self.unsup_rcnn_cls(
                                bbox_results["cls_score"],
                                bbox_targets[0],
                                bbox_targets[1],
                                avg_factor=avg_factor
                        )}
            else:
                loss_unreduced = self.unsup_rcnn_cls(
                                    bbox_results["cls_score"],
                                    bbox_targets[0],
                                    weight=None,
                                    avg_factor=None,
                                    reduction_override='none'
                                )
                loss_unreduced_pos = loss_unreduced[~neg_inds]
                weighted_loss_unreduced_pos = loss_unreduced_pos * bbox_targets[1][~neg_inds, None]
                r = loss_unreduced_pos.sum() / (weighted_loss_unreduced_pos.sum() + 1e-8)
                loss_unreduced = loss_unreduced * bbox_targets[1][..., None]
                loss_unreduced[~neg_inds] = weighted_loss_unreduced_pos * r
                loss = {"loss_cls": loss_unreduced.sum()/avg_factor}
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

    def extract_teacher_info(self, feat, img_metas, **kwargs):
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
        teacher_info = {}
        teacher_info["det_bboxes"] = det_bboxes
        teacher_info["det_labels"] = det_labels
        pseudo_label_weights = [torch.ones_like(label).float() for label in det_labels]
        pseudo_label_weights = torch.cat(pseudo_label_weights, dim=0)
        num_labels_per_img = [label.shape[0] for label in det_labels]
        pseudo_label_weights, pseudo_label_weights_unclamped = self.reweight_label_weights(
            torch.cat(det_labels, dim=0),
            pseudo_label_weights,
            img_metas,
            num_labels_per_img=num_labels_per_img,
            return_unclamped=True
        )
        pseudo_label_weights_list = torch.split(pseudo_label_weights, num_labels_per_img)
        pseudo_label_weights_unclamped_list = torch.split(pseudo_label_weights_unclamped, num_labels_per_img)
        teacher_info["transform_matrix"] = [
            torch.from_numpy(meta["transform_matrix"]).float().to(feat[0][0].device).requires_grad_(False)
            for meta in img_metas
        ]
        teacher_info["img_metas"] = img_metas
        teacher_info["pseudo_label_weights"] = pseudo_label_weights_list
        teacher_info["pseudo_label_weights_unclamped"] = pseudo_label_weights_unclamped_list
        if (not self.no_unlabeled_gt) and "gt_obboxes" in kwargs:
            teacher_info["gt_obboxes"] = kwargs["gt_obboxes"]
            teacher_info["gt_labels"] = kwargs["gt_labels"]
            pseudo_label_weights_gt = [torch.ones_like(label).float() for label in teacher_info["gt_labels"]]
            pseudo_label_weights_gt = torch.cat(pseudo_label_weights_gt, dim=0)
            num_labels_per_img = [label.shape[0] for label in teacher_info["gt_labels"]]
            pseudo_label_weights_gt, pseudo_label_weights_gt_unclamped = self.reweight_label_weights(
                torch.cat(teacher_info["gt_labels"], dim=0),
                pseudo_label_weights_gt,
                img_metas,
                num_labels_per_img=num_labels_per_img,
                return_unclamped=True
            )
            pseudo_label_weights_gt_list = torch.split(pseudo_label_weights_gt, num_labels_per_img)
            pseudo_label_weights_gt_unclamped_list = torch.split(pseudo_label_weights_gt_unclamped, num_labels_per_img)
            teacher_info["pseudo_label_weights_gt"] = pseudo_label_weights_gt_list
            teacher_info["pseudo_label_weights_gt_unclamped"] = pseudo_label_weights_gt_unclamped_list
        
        return det_bboxes, det_labels, proposals, feat, cls_scores_list, teacher_info

    def reweight_label_weights(self, labels, label_weights, img_metas, num_labels_per_img=None, return_unclamped=False):
        # Ensure the input tensors and lists have expected shapes and types
        assert labels.ndim == 1, "labels should be a 1D tensor"
        assert label_weights.ndim == 1, "label_weights should be a 1D tensor"
        assert isinstance(img_metas, list), "img_metas should be a list"

        num_imgs = len(img_metas)  # Number of images
        if num_labels_per_img is None:
            num_labels_per_img = [labels.shape[0] // num_imgs] * num_imgs  # Assumes evenly distributed labels per image

        # Initialize the result tensor and copy the original weights.
        reweighted_label_weights = torch.empty_like(label_weights)
        if return_unclamped:
            reweighted_label_weights_unclamped = torch.empty_like(label_weights)

        for img_idx in range(num_imgs):
            start_idx = img_idx * num_labels_per_img[img_idx]
            end_idx = (img_idx + 1) * num_labels_per_img[img_idx]

            # Extract the labels and their corresponding weights for the current image
            current_labels = labels[start_idx:end_idx]
            current_weights = label_weights[start_idx:end_idx].clone()
            if return_unclamped:
                current_weights_unclamped = current_weights.clone()

            # Retrieve the Ground Sample Distance (GSD) value for the current image
            gsd_value_ori = img_metas[img_idx]['img_info']['gsd']
            if 'scale_factor' in  img_metas[img_idx]:
                gsd_value_ori /= img_metas[img_idx]['scale_factor'].mean()
            elif 'img_shape' in img_metas[img_idx]:
                scale_factor = (float(img_metas[img_idx]['img_shape'][0])/img_metas[img_idx]['img_info']['height']+
                                float(img_metas[img_idx]['img_shape'][1])/img_metas[img_idx]['img_info']['width'])/2.0
                gsd_value_ori /= scale_factor
            gsd_value = math.log((self.lmbda + gsd_value_ori), self.base)
            
            # 仅针对正样本（label < self.num_classes）进行 reweighting
            pos_mask = current_labels < self.num_classes
            if pos_mask.any():
                # 获取正样本对应的标签及其统计量
                pos_labels = current_labels[pos_mask]
                valid_mask = self.gsd_W_per_cls[pos_labels] > 1e-6
                mean = self.gsd_mean_per_cls[pos_labels][valid_mask]
                variance = self.gsd_M2_per_cls[pos_labels][valid_mask] / self.gsd_W_per_cls[pos_labels][valid_mask]
                # 计算 reweight factor，注意 gaussian_pdf_confidence 返回归一化到[0, 1]的置信度
                reweight_factors = self.gaussian_pdf_confidence(gsd_value, mean, torch.clamp(variance, min=0.01))
                # 修改正样本的权重
                # current_weights[pos_mask][valid_mask] *= reweight_factors # huge mistake!
                pos_indices = pos_mask.nonzero().squeeze(-1)
                current_weights[pos_indices[valid_mask]] *= torch.clamp(reweight_factors, min=0.1)
                current_weights[pos_indices[~valid_mask]] *= 0.1
                if return_unclamped:
                    current_weights_unclamped[pos_indices[valid_mask]] *= reweight_factors
                    current_weights_unclamped[pos_indices[~valid_mask]] *= 0.0

            # 更新 reweighted_label_weights 中对应部分，负样本保持原权重
            reweighted_label_weights[start_idx:end_idx] = current_weights
            if return_unclamped:
                reweighted_label_weights_unclamped[start_idx:end_idx] = current_weights_unclamped
        
        if return_unclamped:
            return reweighted_label_weights, reweighted_label_weights_unclamped

        return reweighted_label_weights
    
    def obtain_kld_image_level(self, det_labels_list):
        kld = self.gaussian_kld_divergence(self.gsd_mean_per_cls, (self.gsd_M2_per_cls / self.gsd_W_per_cls),
                                                        self.gsd_mean_per_cls_unlabeled, 
                                                        (self.gsd_M2_per_cls_unlabeled / self.gsd_W_per_cls_unlabeled),
                                                        normalize=False)
        normalized_kld = torch.full_like(kld, -1)
        normalized_kld[kld>=0] = torch.log1p(kld[kld>=0])
        kld_image = [calculate_average_metric_for_labels(det_labels, kld) for det_labels in det_labels_list]
        normalized_kld_image = [calculate_average_metric_for_labels(det_labels, normalized_kld) for det_labels in det_labels_list]
        
        return kld_image, normalized_kld_image
    
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

    @staticmethod
    def gaussian_pdf_confidence(x, mean, variance):
        return torch.exp( - ((x - mean) ** 2) / (2 * variance) )

    @staticmethod
    def gaussian_kld_divergence(mean_1, variance_1, mean_2, variance_2, normalize=True):
        """
        Calculate KL divergence between two univariate Gaussians with:
        - Automatic handling of zero/NaN variances (returns 0 for those positions)
        - ln(1+KLD) normalization for valid elements
        
        Args:
            mean_1: Mean of first Gaussian (torch.Tensor)
            variance_1: Variance of first Gaussian (torch.Tensor)
            mean_2: Mean of second Gaussian (torch.Tensor)
            variance_2: Variance of second Gaussian (torch.Tensor)
        
        Returns:
            Normalized KL divergence with zeros for invalid variances (torch.Tensor)
        """
        # Create validity mask (positive, finite variances)
        valid_mask = ((~torch.isnan(variance_1)) & 
                    (~torch.isnan(variance_2)))
        
        # Initialize output with zeros
        device = mean_1.device
        output = torch.full_like(mean_1, fill_value=-1, device=device) # -1 for invalid elements
        
        # Only compute for valid elements
        if valid_mask.any():
            # Extract valid elements
            m1 = mean_1[valid_mask]
            v1 = variance_1[valid_mask].clamp(min=0.01)
            m2 = mean_2[valid_mask]
            v2 = variance_2[valid_mask].clamp(min=0.01)
            
            # Compute KL divergence terms
            log_ratio = torch.log(v2 / v1)
            squared_diff = (m1 - m2).pow(2)
            term = (v1 + squared_diff) / v2
            
            # Compute raw KL divergence
            kld = 0.5 * (log_ratio + term - 1)
            
            if normalize:
                # Apply ln(1 + KLD) normalization
                output[valid_mask] = torch.log1p(kld)
            else:
                output[valid_mask] = kld
        
        return output