import torch
import os
import random

from mmcv.runner import force_fp32, get_dist_info
from mmdet.core import multi_apply, arb2roi, bbox2type, get_bbox_dim
from mmdet.models import BaseDetector, TwoStageDetector, DETECTORS, build_detector
from mmdet.models.builder import build_loss
from mmdet.models.losses import accuracy

from ssod.utils.structure_utils import dict_split, weighted_loss
from ssod.utils import log_every_n
from ssod.models.utils import filter_invalid, process_visualization, visualize_images

from .multi_stream_detector import MultiStreamDetector


@DETECTORS.register_module()
class UnbiasedTeacherV2(MultiStreamDetector):
    def __init__(self, model: dict, train_cfg=None, test_cfg=None):
        super(UnbiasedTeacherV2, self).__init__(
            dict(teacher=build_detector(model), student=build_detector(model)),
            train_cfg=train_cfg,
            test_cfg=test_cfg,
        )
        self.freeze("teacher")
        if self.train_cfg is not None:
            self.unsup_weight = self.train_cfg.unsup_weight
            self.use_hbb_rpn = self.train_cfg.get("use_hbb_rpn", False)
            cls_loss_cfg = self.train_cfg.get("unsup_rcnn_cls", None)
            if cls_loss_cfg:
                self.unsup_loss_cls = build_loss(cls_loss_cfg)
            reg_loss_cfg = self.train_cfg.get("unsup_rcnn_reg", None)
            if reg_loss_cfg:
                self.unsup_loss_bbox = build_loss(reg_loss_cfg)
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
        # sort the teacher and student input to avoid some bugs
        tnames = [meta["filename"] for meta in teacher_data["img_metas"]]
        snames = [meta["filename"] for meta in student_data["img_metas"]]
        tidx = [tnames.index(name) for name in snames]
        with torch.no_grad():
            teacher_input = {
                "img": teacher_data["img"][torch.Tensor(tidx).to(teacher_data["img"].device).long()],
                "img_metas": [teacher_data["img_metas"][idx] for idx in tidx],
                "proposals": [teacher_data["proposals"][idx] for idx in tidx]
                            if "proposals" in teacher_data
                            and teacher_data["proposals"] is not None else None,
            }
            if "gt_obboxes" in teacher_data and teacher_data["gt_obboxes"] is not None:
                teacher_input["gt_obboxes"] = [teacher_data["gt_obboxes"][idx] for idx in tidx]
                teacher_input["gt_labels"] = [teacher_data["gt_labels"][idx] for idx in tidx]
            teacher_info = self.extract_teacher_info(**teacher_input)
        student_info = self.extract_student_info(**student_data)
        
        if visualize:
            process_visualization(teacher_info, student_info, teacher_data, student_data,
                                  "data/DOTA/show_data_trans_softT_0613")
            
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
            pseudo_bboxes_std = [bbox_std[mask] for bbox_std, mask 
                                 in zip(teacher_info["det_bboxes_std"], valid_masks)]
            pseudo_labels = [label[mask] for label, mask in zip(teacher_info["det_labels"], valid_masks)]
                
        # visualize_images(
        #     teacher_info["img"], teacher_info["img_metas"],
        #     [tbb[:, :6] for tbb in teacher_info["det_bboxes"]], teacher_info["det_labels"], "pseudo_teacher",
        #     "data/DOTA/show_data_trans_softT_midprocess"
        # )
        # visualize_images(
        #     student_info["img"], student_info["img_metas"],
        #     [pbb[:, :6] for pbb in pseudo_bboxes], pseudo_labels, "pseudo_student",
        #     "data/DOTA/show_data_trans_softT_midprocess"
        # )
        # visualize_images(
        #     student_info["img"], student_info["img_metas"],
        #     student_info['gt_obboxes'], 
        #     student_info['gt_labels'],
        #     "gt_student", "data/DOTA/show_data_trans_softT_midprocess"
        # )

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
            
            # visualize_images(
            #     teacher_info["img"], teacher_info["img_metas"],
            #     [bbox2type(tpp[:, :4], "obb") if self.use_hbb_rpn else tpp[:, :5]
            #      for tpp in teacher_info["proposals"]], 
            #     None,
            #     "teacher_proposal", "data/DOTA/show_data_trans_softT_midprocess"
            # )
            # visualize_images(
            #     student_info["img"], student_info["img_metas"],
            #     [bbox2type(pp[:, :4], "obb") if self.use_hbb_rpn else pp[:, :5]
            #      for pp in proposals], 
            #     None,
            #     "trans_student_proposal", "data/DOTA/show_data_trans_softT_midprocess"
            # )
            # visualize_images(
            #     student_info["img"], student_info["img_metas"],
            #     [bbox2type(spp[:, :4], "obb") if self.use_hbb_rpn else spp[:, :5]
            #      for spp in student_info["proposals"]], 
            #     None,
            #     "student_proposal", "data/DOTA/show_data_trans_softT_midprocess"
            # )
            
        else:
            proposals = student_info["proposals"]

        original_loss_cls, original_loss_bbox = self.backup_loss_settings()
        loss.update(
            self.unsup_rcnn_loss(
                student_info["backbone_feature"],
                student_info["img_metas"],
                proposals,
                pseudo_bboxes,
                pseudo_bboxes_std,
                pseudo_labels,
                student_info=student_info,
            )
        )
        self.restore_loss_settings(original_loss_cls, original_loss_bbox)
        return loss

    def rpn_loss(self,
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

    def unsup_rcnn_loss(self,
        feat,
        img_metas,
        proposal_list,
        pseudo_bboxes,
        pseudo_bboxes_std,
        pseudo_labels,
        **kwargs,
    ):
        gt_obboxes_with_score, gt_labels, _, idx = multi_apply(
            filter_invalid,
            [bbox for bbox in pseudo_bboxes],
            pseudo_labels,
            [bbox[:, 5] for bbox in pseudo_bboxes],
            thr=self.train_cfg.cls_reg_pseudo_threshold,
            return_inds=True,
        )
        gt_obboxes = [bbox[:, :5] for bbox in gt_obboxes_with_score]
        gt_scores = [bbox[:, 5] for bbox in gt_obboxes_with_score]
        gt_loc_stds = [bbox_std[mask] for bbox_std, mask in zip(pseudo_bboxes_std, idx)]
        log_every_n(
            {"rcnn_cls_gt_num": sum([len(bbox) for bbox in gt_obboxes]) / len(gt_obboxes)}
        )
        sampling_results, gt_bboxes = self.get_sampling_result(
            img_metas,
            proposal_list,
            gt_obboxes,
            gt_labels,
            pseudo_confids=gt_scores,
            pseudo_loc_stds=gt_loc_stds
        )
        selected_bboxes = [res.bboxes[:, :5] for res in sampling_results] # still compatiable for hbb proposals
        # rois = bbox2roi(selected_bboxes)
        rois = arb2roi(selected_bboxes, bbox_type=self.student.roi_head.bbox_head.start_bbox_type)
        losses = dict()
        bbox_results = self.student.roi_head._bbox_forward_train(
            feat, sampling_results,
            gt_bboxes, gt_labels,
            rois=rois,
            gt_loc_std=torch.cat([res.gt_loc_stds for res in sampling_results], dim=0) 
                                    if hasattr(sampling_results[0], 'gt_loc_stds') else None,
            img_metas=img_metas)
        losses.update(bbox_results['loss_bbox'])
        return losses

    def get_sampling_result(
        self,
        img_metas,
        proposal_list,
        gt_obboxes,
        gt_labels,
        pseudo_confids=None,
        pseudo_loc_stds=None,
        gt_bboxes_ignore=None,
        **kwargs,
    ):
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
            
            # Handle case where we need to work with oriented bounding boxes (OBB)
            if self.use_hbb_rpn:
                if gt_obboxes[i].numel() == 0:
                    sampling_result.pos_gt_bboxes = gt_obboxes[i].new_zeros(
                        (0, gt_obboxes[0].size(-1)))
                else:
                    sampling_result.pos_gt_bboxes = gt_obboxes[i][sampling_result.pos_assigned_gt_inds, :]

            # Incorporate pseudo-label confidence scores and uncertainty
            if pseudo_confids is not None:
                sampling_result.gt_confidences = pseudo_confids[i][sampling_result.pos_assigned_gt_inds]
            
            if pseudo_loc_stds is not None:
                sampling_result.gt_loc_stds = pseudo_loc_stds[i][sampling_result.pos_assigned_gt_inds, :]

            sampling_results.append(sampling_result)
        return sampling_results, gt_bboxes

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
       
        det_bboxes, det_labels, det_bboxes_std = self.teacher.roi_head.simple_test_bboxes(
            feat, img_metas, proposal_list, 
            self.teacher.test_cfg.rcnn, rescale=False,
            return_var=True
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
        det_bboxes, det_labels, _, idx = list(
            zip(
                *[
                    filter_invalid(
                        proposal,
                        proposal_label,
                        proposal[:, -1],
                        thr=thr,
                        min_size=self.train_cfg.min_pseduo_box_size,
                        return_inds=True
                    )
                    for proposal, proposal_label in zip(
                        det_bboxes, det_labels
                    )
                ]
            )
        )
        teacher_info["det_bboxes"] = det_bboxes
        teacher_info["det_labels"] = det_labels
        teacher_info["det_bboxes_std"] = [det_bbox_std[mask] for det_bbox_std, mask 
                                          in zip(det_bboxes_std, idx)]
        teacher_info["transform_matrix"] = [
            torch.from_numpy(meta["transform_matrix"]).float().to(feat[0][0].device).requires_grad_(False)
            for meta in img_metas
        ]
        teacher_info["img_metas"] = img_metas
        if "gt_obboxes" in kwargs:
            teacher_info["gt_obboxes"] = kwargs["gt_obboxes"]
            teacher_info["gt_labels"] = kwargs["gt_labels"]
        return teacher_info

    def backup_loss_settings(self):
        original_loss_cls = self.student.roi_head.bbox_head.loss_cls
        original_loss_bbox = self.student.roi_head.bbox_head.loss_bbox
        self.student.roi_head.bbox_head.loss_cls = self.unsup_loss_cls
        self.student.roi_head.bbox_head.loss_bbox = self.unsup_loss_bbox
        return original_loss_cls, original_loss_bbox
    
    def restore_loss_settings(self, original_loss_cls, original_loss_bbox):
        self.student.roi_head.bbox_head.loss_cls = original_loss_cls
        self.student.roi_head.bbox_head.loss_bbox = original_loss_bbox