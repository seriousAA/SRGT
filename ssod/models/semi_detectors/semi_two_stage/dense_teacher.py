import torch
import os

from mmdet.core import multi_apply, OBBOverlaps, build_assigner, arb2roi, bbox2type, get_bbox_dim
from mmdet.models import BaseDetector, TwoStageDetector, DETECTORS, build_detector
from mmdet.models.builder import build_loss
from mmdet.models.losses import accuracy

from ssod.utils.structure_utils import dict_split, weighted_loss
from ssod.utils import log_every_n
from ssod.models.utils import filter_invalid, process_visualization, visualize_images

from .multi_stream_detector import MultiStreamDetector


@DETECTORS.register_module()
class DenseTeacher(MultiStreamDetector):
    def __init__(self, model: dict, train_cfg=None, test_cfg=None):
        super(DenseTeacher, self).__init__(
            dict(teacher=build_detector(model), student=build_detector(model)),
            train_cfg=train_cfg,
            test_cfg=test_cfg,
        )
        self.freeze("teacher")
        if self.train_cfg is not None:
            self.unsup_weight = self.train_cfg.unsup_weight
            self.use_hbb_rpn = self.train_cfg.get("use_hbb_rpn", False)
            semi_loss_cfg = self.train_cfg.get("semi_loss", dict(type='RotatedDTLoss'))
            assert semi_loss_cfg.type in ["RotatedDTLoss", "RotatedTSOTDTLoss"]
            self.semi_loss = build_loss(semi_loss_cfg)
            assert self.train_cfg.use_teacher_proposal, \
                "Student model using teacher proposals is required for dense teacher."

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
        if "sup" in data_groups:
            gt_bboxes = data_groups["sup"]["gt_bboxes"]
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
        # sort the teacher and student input to avoid some bugs
        tnames = [meta["filename"] for meta in teacher_data["img_metas"]]
        snames = [meta["filename"] for meta in student_data["img_metas"]]
        tidx = [tnames.index(name) for name in snames]
        with torch.no_grad():
            teacher_info = {
                "img": teacher_data["img"][torch.Tensor(tidx).to(teacher_data["img"].device).long()],
                "img_metas": [teacher_data["img_metas"][idx] for idx in tidx],
                "proposals": [teacher_data["proposals"][idx] for idx in tidx]
                            if "proposals" in teacher_data
                            and teacher_data["proposals"] is not None else None,
            }
            if "gt_obboxes" in teacher_data and teacher_data["gt_obboxes"] is not None:
                teacher_info["gt_obboxes"] = [teacher_data["gt_obboxes"][idx] for idx in tidx]
                teacher_info["gt_labels"] = [teacher_data["gt_labels"][idx] for idx in tidx]
            teacher_info = self.extract_teacher_info(**teacher_info)
        student_info = self.extract_student_info(**student_data)
        
        if visualize:
            process_visualization(teacher_info, student_info, teacher_data, student_data,
                                  "data/DOTA/show_data_trans_dt_0613")
            
        return self.compute_pseudo_label_loss(student_info, teacher_info)

    def compute_pseudo_label_loss(self, student_info, teacher_info):
        with torch.no_grad():
            M = self._get_trans_mat(
                teacher_info["transform_matrix"], student_info["transform_matrix"]
            )

            pseudo_bboxes, valid_masks = self._transform_bbox(
                'obb', teacher_info["det_bboxes"],
                M,
                [meta["img_shape"] for meta in student_info["img_metas"]],
            )
            pseudo_labels = [label[mask] for label, mask in zip(teacher_info["det_labels"], valid_masks)]

        loss = {}
        rpn_loss, proposal_list = self.rpn_loss(
            student_info["rpn_out"],
            pseudo_bboxes,
            student_info["img_metas"],
            student_info=student_info,
        )
        loss.update(rpn_loss)
        if proposal_list is not None:
            student_info["proposals"] = proposal_list
        if self.train_cfg.use_teacher_proposal:
            with torch.no_grad():
                proposals, _ = self._transform_bbox(
                    'hbb' if self.use_hbb_rpn else 'obb', teacher_info["proposals"],
                    M,
                    [meta["img_shape"] for meta in student_info["img_metas"]],
                )
        else:
            proposals = student_info["proposals"]
            
        rois = arb2roi(proposals, bbox_type=self.student.roi_head.bbox_head.start_bbox_type)
        bbox_results = self.student.roi_head._bbox_forward(student_info["backbone_feature"], rois)
        student_info["preds_cls"] = bbox_results["cls_score"]
        student_info["preds_bbox"] = bbox_results["bbox_pred"]
        
        with torch.no_grad():
            M_ = self._get_trans_mat(student_info["transform_matrix"], teacher_info["transform_matrix"])
            aligned_proposals, _ = self._transform_bbox(
                'hbb' if self.use_hbb_rpn else 'obb', proposals,
                M_
            )
        rois_ = arb2roi(aligned_proposals, bbox_type=self.teacher.roi_head.bbox_head.start_bbox_type)
        bbox_results_ = self.teacher.roi_head._bbox_forward(teacher_info["backbone_feature"], rois_)
        teacher_info["preds_cls"] = bbox_results_["cls_score"]
        teacher_info["preds_bbox"] = bbox_results_["bbox_pred"]
        
        num_per_img = [p.shape[0] for p in proposals]
        loss.update(
            self.semi_loss(teacher_info, student_info, num_per_img, aligned_proposals=aligned_proposals)
        )
        return loss

    def rpn_loss(
        self,
        rpn_out,
        pseudo_bboxes,
        img_metas,
        gt_bboxes_ignore=None,
        student_info=None,
        **kwargs,
    ):
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

    def extract_student_info(self, img, img_metas, proposals=None, **kwargs):
        student_info = {}
        student_info["img"] = img
        feat = self.student.extract_feat(img)
        student_info["backbone_feature"] = feat
        
        rpn_out = self.student.rpn_head(feat)
        student_info["rpn_out"] = list(rpn_out)
        
        student_info["img_metas"] = img_metas
        student_info["proposals"] = proposals
        student_info["transform_matrix"] = [
            torch.from_numpy(meta["transform_matrix"]).float().to(feat[0][0].device).requires_grad_(False)
            for meta in img_metas
        ]
        if "gt_obboxes" in kwargs:
            student_info["gt_obboxes"] = kwargs["gt_obboxes"]
            student_info["gt_labels"] = kwargs["gt_labels"]
        return student_info

    def extract_teacher_info(self, img, img_metas, proposals=None, **kwargs):
        teacher_info = {}
        teacher_info["img"] = img
        feat = self.teacher.extract_feat(img)
        teacher_info["backbone_feature"] = feat
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
        teacher_info["proposals"] = proposal_list

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
        teacher_info["det_bboxes"] = det_bboxes
        teacher_info["det_labels"] = det_labels
        teacher_info["transform_matrix"] = [
            torch.from_numpy(meta["transform_matrix"]).float().to(feat[0][0].device).requires_grad_(False)
            for meta in img_metas
        ]
        teacher_info["img_metas"] = img_metas
        if "gt_obboxes" in kwargs:
            teacher_info["gt_obboxes"] = kwargs["gt_obboxes"]
            teacher_info["gt_labels"] = kwargs["gt_labels"]
        return teacher_info
