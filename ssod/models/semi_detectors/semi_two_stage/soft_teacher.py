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
class SoftTeacher(MultiStreamDetector):
    def __init__(self, model: dict, train_cfg=None, test_cfg=None):
        super(SoftTeacher, self).__init__(
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

        loss.update(
            self.unsup_rcnn_cls_loss(
                student_info["backbone_feature"],
                student_info["img_metas"],
                proposals,
                pseudo_bboxes,
                pseudo_labels,
                teacher_info["transform_matrix"],
                student_info["transform_matrix"],
                teacher_info["img_metas"],
                teacher_info["backbone_feature"],
                student_info=student_info,
                teacher_info=teacher_info,
            )
        )
        loss.update(
            self.unsup_rcnn_reg_loss(
                student_info["backbone_feature"],
                student_info["img_metas"],
                proposals,
                pseudo_bboxes,
                pseudo_labels,
                student_info=student_info,
            )
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

    def unsup_rcnn_cls_loss(
        self,
        feat,
        img_metas,
        proposal_list,
        pseudo_bboxes,
        pseudo_labels,
        teacher_transMat,
        student_transMat,
        teacher_img_metas,
        teacher_feat,
        student_info=None,
        **kwargs,
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
        sampling_results, gt_bboxes = self.get_sampling_result(
            img_metas,
            proposal_list,
            gt_obboxes,
            gt_labels,
        )
        selected_bboxes = [res.bboxes[:, :5] for res in sampling_results] # still compatiable for hbb proposals
        # rois = bbox2roi(selected_bboxes)
        rois = arb2roi(selected_bboxes, bbox_type=self.student.roi_head.bbox_head.start_bbox_type)
        bbox_results = self.student.roi_head._bbox_forward(feat, rois)
        bbox_targets = self.student.roi_head.bbox_head.get_targets(
            # 'gt_bboxes' should be 'hbb', but actually it won't be used in the training targets obtaining
            sampling_results, gt_bboxes, gt_labels, self.student.train_cfg.rcnn
        )
        
        with torch.no_grad():
            M = self._get_trans_mat(student_transMat, teacher_transMat)
            aligned_proposals, _ = self._transform_bbox(
                'hbb' if self.use_hbb_rpn else 'obb', selected_bboxes,
                M
            )
            # inference the selected proposals on teacher model, extract the bg score as the label weight
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
            bbox_targets[1][neg_inds] = bg_score[neg_inds]
            
        # pos_nums = [res.pos_inds.shape[0] for res in sampling_results]
        # visualize_images(
        #     student_info["img"], student_info["img_metas"],
        #     [bbox2type(sbb[:pos_num, :4], "obb") if self.use_hbb_rpn else sbb[:pos_num, :5]
        #         for pos_num, sbb in zip(pos_nums, selected_bboxes)], 
        #     None,
        #     "pos_student_proposal", "data/DOTA/show_data_trans_softT_midprocess"
        # )
        # visualize_images(
        #     kwargs["teacher_info"]["img"], kwargs["teacher_info"]["img_metas"],
        #     [bbox2type(app[:pos_num, :4], "obb") if self.use_hbb_rpn else app[:pos_num, :5]
        #         for pos_num, app in zip(pos_nums, aligned_proposals)], 
        #     None,
        #     "al_pos_teacher_proposal", "data/DOTA/show_data_trans_softT_midprocess"
        # )
        # visualize_images(
        #     student_info["img"], student_info["img_metas"],
        #     [bbox2type(sbb[pos_num:, :4], "obb") if self.use_hbb_rpn else sbb[pos_num:, :5]
        #         for pos_num, sbb in zip(pos_nums, selected_bboxes)], 
        #     None,
        #     "neg_student_proposal", "data/DOTA/show_data_trans_softT_midprocess", 'red'
        # )
        # visualize_images(
        #     kwargs["teacher_info"]["img"], kwargs["teacher_info"]["img_metas"],
        #     [bbox2type(app[pos_num:, :4], "obb") if self.use_hbb_rpn else app[pos_num:, :5]
        #         for pos_num, app in zip(pos_nums, aligned_proposals)], 
        #     None,
        #     "al_neg_teacher_proposal", "data/DOTA/show_data_trans_softT_midprocess", 'red'
        # )
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

    def get_sampling_result(
        self,
        img_metas,
        proposal_list,
        gt_obboxes,
        gt_labels,
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
            if self.use_hbb_rpn:
                if gt_obboxes[i].numel() == 0:
                    sampling_result.pos_gt_bboxes = gt_obboxes[i].new_zeros(
                        (0, gt_obboxes[0].size(-1)))
                else:
                    sampling_result.pos_gt_bboxes = \
                            gt_obboxes[i][sampling_result.pos_assigned_gt_inds, :]

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
        reg_unc = self.compute_uncertainty_with_aug(
            feat, img_metas, det_bboxes, det_labels
        )
        det_bboxes = [
            torch.cat([bbox, unc], dim=-1) for bbox, unc in zip(det_bboxes, reg_unc)
        ]
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
