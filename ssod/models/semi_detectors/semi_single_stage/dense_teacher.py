import torch
import os
import torch.nn.functional as F
import cv2
import numpy as np

from mmdet.core import multi_apply, distance2obb, obb2distance, obb2poly, mintheta_obb
from mmdet.ops import obb_overlaps
from mmcv.ops import points_in_polygons
from mmdet.models import DETECTORS, build_detector
from mmdet.models.builder import build_loss
from mmdet.models.losses import accuracy

from ssod.utils.structure_utils import dict_split, weighted_loss
from ssod.utils import log_every_n
from ssod.models.utils import (filter_invalid, visualize_points, visualize_images,
                               visualize_images_with_points)
from ssod.models.losses.utils import OT_Loss

from .multi_stream_st_detector import MultiStreamSTDetector


@DETECTORS.register_module()
class STDenseTeacher(MultiStreamSTDetector):
    def __init__(self, model: dict, train_cfg=None, test_cfg=None):
        super(STDenseTeacher, self).__init__(
            dict(teacher=build_detector(model), student=build_detector(model)),
            train_cfg=train_cfg,
            test_cfg=test_cfg,
        )
        self.freeze("teacher")
        if self.train_cfg is not None:
            self.unsup_weight = self.train_cfg.unsup_weight
            self.teacher_dets_for_mask = self.train_cfg.get("dets_for_mask", False)
            self.teacher_dets2mask_threshold = self.train_cfg.get("dets2mask_threshold", 0.5)
            semi_loss_cfg = self.train_cfg.get("semi_loss")
            assert semi_loss_cfg.type in ["RotatedDTLoss", "RotatedMCLLoss"]
            self.semi_loss = build_loss(semi_loss_cfg)
            self.dynamic_raw_type=self.train_cfg.get("dynamic_raw_type")
            self.with_gc_loss = self.train_cfg.get("with_gc_loss", False)
            if self.with_gc_loss:
                assert self.teacher_dets_for_mask, "GC loss requires teacher detections for masking"
                ot_loss_cfg = self.train_cfg.get("ot_loss_cfg", {})
                self.ot_type = ot_loss_cfg.pop('ot_type', 'ot_loss_norm')
                assert self.ot_type in ['ot_loss_norm', 'ot_ang_loss_norm']
                self.ot_weight = ot_loss_cfg.pop('loss_weight', 1.)
                self.cost_type = ot_loss_cfg.pop('cost_type', 'all')
                assert self.cost_type in ['all', 'dist', 'score']
                self.clamp_ot = ot_loss_cfg.pop('clamp_ot', False)
                self.gc_loss = OT_Loss(**ot_loss_cfg)

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
            }
            if "gt_obboxes" in teacher_data and teacher_data["gt_obboxes"] is not None:
                teacher_info["gt_obboxes"] = [teacher_data["gt_obboxes"][idx] for idx in tidx]
                teacher_info["gt_labels"] = [teacher_data["gt_labels"][idx] for idx in tidx]
            teacher_info = self.extract_model_info(self.teacher, **teacher_info,
                                                   get_dets=self.teacher_dets_for_mask)
        student_info = self.extract_model_info(self.student, **student_data)
        
        return self.compute_pseudo_label_loss(student_info, teacher_info)

    def compute_pseudo_label_loss(self, student_info, teacher_info):
        tea_featmap_sizes = [pred_layer.shape[-2:] for pred_layer in teacher_info["cls_scores"]]
        tea_mlvl_points = self.teacher.bbox_head.get_points(tea_featmap_sizes, 
                                                            teacher_info["bbox_preds"][0].dtype, 
                                                            teacher_info["bbox_preds"][0].device,
                                                            no_flatten=True)
        stu_featmap_sizes = [pred_layer.shape[-2:] for pred_layer in student_info["cls_scores"]]
        stu_mlvl_points = self.student.bbox_head.get_points(stu_featmap_sizes, 
                                                            student_info["bbox_preds"][0].dtype, 
                                                            student_info["bbox_preds"][0].device,
                                                            no_flatten=True)

        teacher_predictions = (
            teacher_info["cls_scores"],
            teacher_info["bbox_preds"],
            teacher_info["theta_preds"],
            teacher_info["centerness"]
        )
        teacher_predictions = self.reshape_batched_predictions(teacher_predictions,
                                                               to_permute=True)
        student_predictions = (
            student_info["cls_scores"],
            student_info["bbox_preds"],
            student_info["theta_preds"],
            student_info["centerness"]
        )
        student_predictions = self.reshape_batched_predictions(student_predictions,
                                                               to_permute=True)
        
        with torch.no_grad():
            M_stu_tea = self._get_trans_mat(
                student_info["transform_matrix"], teacher_info["transform_matrix"]
            )
            M_tea_stu = self._get_trans_mat(
                teacher_info["transform_matrix"], student_info["transform_matrix"]
            )
            # Get valid student masks and corresponding teacher indices
            valid_student_masks, teacher_mapping_indices = \
                self.get_valid_mapping_masks_indices(stu_mlvl_points, tea_mlvl_points, M_stu_tea)

        if self.teacher_dets_for_mask:
            # Map the teacher detections onto the first featmap layer to 
            # further mask the student grid points
            teacher_dets_masks = self.teacher_dets_to_masks(
                teacher_info["det_bboxes"],
                [img.shape[-2:] for img in student_info["img"]],
                M_tea_stu
            )
            # If use the teacher detection masks, then only consider predictions and targets
            # on the first student featmap level for each unlabeled image in the batch
            student_predictions = [[preds[:1] for preds in batch_preds] 
                                    for batch_preds in student_predictions]
            teacher_predictions = [[preds[:1] for preds in batch_preds] 
                                    for batch_preds in teacher_predictions]
            valid_student_masks = [[valid_student_masks[i][0] & teacher_dets_masks[i]] 
                                   for i in range(len(valid_student_masks))]
            teacher_mapping_indices = [indices[:1] for indices in teacher_mapping_indices]
            stu_mlvl_points = stu_mlvl_points[:1]
            tea_mlvl_points = tea_mlvl_points[:1]
                
        batched_valid_stu_preds, batched_valid_pred_targets, \
            batched_valid_stu_bboxes, batched_pseudo_obbox_targets, batched_valid_tea_bboxes, \
                batched_valid_stu_points, batched_valid_tea_points, batched_valid_strides = \
                    self.get_valid_preds_targets(
                        student_predictions, teacher_predictions,
                        valid_student_masks, teacher_mapping_indices,
                        stu_mlvl_points, tea_mlvl_points, M_tea_stu
                    )

        # with torch.no_grad():
        #     visualize_images(
        #         student_info["img"], student_info["img_metas"],
        #         [t for t in self.map_norm_obbox_to_imgs(
        #             batched_valid_stu_bboxes, batched_valid_stu_points, batched_valid_strides)], 
        #         None,
        #         "student_2", "data/DOTA/show_data_trans_stdt"
        #     )
        #     visualize_images(
        #         student_info["img"], student_info["img_metas"],
        #         [t for t in self.map_norm_obbox_to_imgs(
        #             batched_pseudo_obbox_targets, batched_valid_stu_points, batched_valid_strides)], 
        #         None,
        #         "tea_on_student_2", "data/DOTA/show_data_trans_stdt"
        #     )
        #     visualize_images(
        #         teacher_info["img"], teacher_info["img_metas"],
        #         [t for t in batched_valid_tea_bboxes], 
        #         None,
        #         "teacher_2", "data/DOTA/show_data_trans_stdt"
        #     )
        #     self.visualize_dets_pts_test(
        #         teacher_info, student_info, M_tea_stu, M_stu_tea,
        #         batched_valid_stu_points, batched_valid_tea_points,
        #     )
        
        loss_weight = None
        if self.dynamic_raw_type is not None:
            if self.dynamic_raw_type in ['ang', '10ang', '50ang', '100ang']:
                loss_weight = [torch.abs(t[:, -1, None] - s[:, -1, None]) / np.pi 
                               for t, s in 
                               zip(batched_pseudo_obbox_targets, batched_valid_stu_bboxes)]
                loss_weight = torch.cat(loss_weight, dim=0)
                if self.dynamic_raw_type == '100ang':
                    loss_weight = torch.clamp(100 * loss_weight, 0, 1) + 1
                elif self.dynamic_raw_type == '50ang':
                    loss_weight = torch.clamp(50 * loss_weight, 0, 1) + 1
                elif self.dynamic_raw_type == '10ang':
                    loss_weight = torch.clamp(10 * loss_weight, 0, 1) + 1
                else:
                    loss_weight = loss_weight + 1
            else:
                raise NotImplementedError(f"Not supported dynamic raw weight type: {self.dynamic_raw_type}")
        
        unsup_loss = {}
        unsup_loss.update(self.semi_loss(
            [(tea_preds[0], tea_bboxes, tea_preds[2]) for tea_preds, tea_bboxes 
             in zip(batched_valid_pred_targets, batched_pseudo_obbox_targets)],
            [(stu_preds[0], stu_bboxes, stu_preds[2]) for stu_preds, stu_bboxes 
             in zip(batched_valid_stu_preds, batched_valid_stu_bboxes)],
            [stu_bboxes.shape[0] for stu_bboxes in batched_valid_stu_bboxes],
            loss_weight=loss_weight,
            denorm_tea_bboxes=batched_valid_tea_bboxes, 
            stu_points=batched_valid_stu_points, 
            tea_points=batched_valid_tea_points, 
            valid_strides=batched_valid_strides
        ))
        if self.with_gc_loss:
            unsup_loss.update(self.calculate_gc_ot_loss(
                batched_valid_stu_preds, batched_valid_pred_targets, 
                valid_student_masks))
        if self.teacher_dets_for_mask:
            valid_locs_num = sum([mask[0].sum() for mask in valid_student_masks]) / \
                                len(valid_student_masks)
            unsup_loss.update({
                'valid_pts_num': valid_locs_num.float()
            })
        return unsup_loss
        

    def extract_model_info(self, model, img, img_metas, **kwargs):
        """Extract relevant model information for semi-supervised training.

        This function can be used for both the teacher and student models
        since the extraction process is the same.

        Args:
            model (nn.Module): The model (either teacher or student) from which to extract information.
            img (Tensor): Input image tensor.
            img_metas (list[dict]): Meta information of the input image.

        Returns:
            dict: A dictionary containing the model's backbone features, head outputs,
                and other relevant information.
        """
        model_info = {}
        img = self.crop_images_to_pad_shape(img, img_metas)
        model_info["img"] = img
        feat = model.extract_feat(img)  # Extract backbone features
        model_info["backbone_feature"] = feat

        # Obtain classification scores, bbox predictions, theta predictions, and centerness predictions directly.
        # Notice that here the tea_bbox_preds will be denormalized back to the image space (*=strides[i])
        # due to the testing mode of the teacher model, while the student model will not do this due to its
        # training mode.
        cls_scores, bbox_preds, theta_preds, centerness = model.bbox_head(feat)
        model_info["cls_scores"] = list(cls_scores)
        model_info["bbox_preds"] = list(bbox_preds)
        model_info["theta_preds"] = list(theta_preds)
        model_info["centerness"] = list(centerness)

        model_info["img_metas"] = img_metas
        model_info["transform_matrix"] = [
            torch.from_numpy(meta["transform_matrix"]).float().to(feat[0][0].device).requires_grad_(False)
            for meta in img_metas
        ]

        # Include ground truth data if provided (for supervised loss calculation)
        if "gt_obboxes" in kwargs:
            model_info["gt_obboxes"] = kwargs["gt_obboxes"]
            model_info["gt_labels"] = kwargs["gt_labels"]
        
        if kwargs.get("get_dets", False):
            dets = model.bbox_head.get_bboxes(
                cls_scores,
                bbox_preds,
                theta_preds,
                centerness,
                img_metas,
                rescale=True
            )
            model_info["det_bboxes"] = [det[0] for det in dets]
            model_info["det_labels"] = [det[1] for det in dets]

        return model_info
    
    def get_valid_preds_targets_single(self,
        single_mlvl_student_preds, single_mlvl_teacher_preds,
        single_valid_student_masks, single_teacher_mapping_indices,
        mlvl_stu_points, mlvl_tea_points, trans_matrix
    ):
        """Process valid predictions and targets for a single image.

        Args:
            single_mlvl_student_preds (tuple): Student predictions 
            (cls_scores, bbox_preds, theta_preds, centerness).
            single_mlvl_teacher_preds (tuple): Teacher predictions 
            (cls_scores, bbox_preds, theta_preds, centerness).
            single_valid_student_masks (list[Tensor]): Valid student masks for 
            each feature level [(H, W), ...].
            single_teacher_mapping_indices (list[Tensor]): Teacher mapping indices 
            for valid student points [(H, W, 2), ...].
            mlvl_stu_points (list[Tensor]): Student points [(H, W, 2), ...] 
            (same for all images in the batch).

        Returns:
            tuple: Flattened valid student predictions, teacher predictions, and decoded bounding boxes.
                - flatten_valid_stu_preds: Tuple of flattened student predictions 
                    (cls_scores, obbox_preds, centerness).
                - flatten_valid_pred_targets: Tuple of flattened pseudo teacher predictions
                    (cls_scores, obbox_preds, centerness).
                - flatten_valid_stu_bboxes: Decoded student bounding boxes (N, 5) on student input images.
                - flatten_pseudo_obbox_targets: Decoded pseudo teacher bounding boxes (N, 5) 
                    mapped onto student input images.
                - flatten_valid_tea_bboxes: Decoded teacher bounding boxes (N, 5) on teacher input images.
                - flatten_valid_stu_points: Flattened student points (N, 2) on student input images.
                - flatten_valid_tea_points: Flattened teacher points (N, 2) on teacher input images.
                - flatten_mlvl_strides: Flattened strides (N, 1) for each feature level.
        """
        flattened_stu_preds = []
        flattened_tea_preds = []
        flattened_stu_points = []
        flattened_tea_points = []
        flattened_strides = []

        for level_idx in range(len(single_mlvl_student_preds[0])):  # Loop through each feature level
            mask = single_valid_student_masks[level_idx]
            stu_cls_pred = single_mlvl_student_preds[0][level_idx][mask]
            stu_bbox_pred = single_mlvl_student_preds[1][level_idx][mask]
            stu_theta_pred = single_mlvl_student_preds[2][level_idx][mask]
            stu_centerness_pred = single_mlvl_student_preds[3][level_idx][mask]

            # Combine the student predictions
            flattened_stu_preds.append((stu_cls_pred, torch.cat([stu_bbox_pred, 
                                            stu_theta_pred], dim=1), stu_centerness_pred))

            # Get corresponding teacher indices and map teacher predictions
            teacher_indices = single_teacher_mapping_indices[level_idx][mask].long()
            tea_cls_pred = single_mlvl_teacher_preds[0][level_idx][teacher_indices[:, 1], 
                                                                      teacher_indices[:, 0]]
            tea_bbox_pred = single_mlvl_teacher_preds[1][level_idx][teacher_indices[:, 1], 
                                                                       teacher_indices[:, 0]]
            tea_theta_pred = single_mlvl_teacher_preds[2][level_idx][teacher_indices[:, 1], 
                                                                        teacher_indices[:, 0]]
            tea_centerness_pred = single_mlvl_teacher_preds[3][level_idx][teacher_indices[:, 1], 
                                                                           teacher_indices[:, 0]]

            # Combine the teacher predictions
            flattened_tea_preds.append((tea_cls_pred, torch.cat([tea_bbox_pred, 
                                        tea_theta_pred], dim=1), tea_centerness_pred))

            # Get the corresponding student points (same for all images in batch)
            stu_points = mlvl_stu_points[level_idx][mask]
            strides = torch.full_like(stu_points[:, 0, None], self.student.bbox_head.strides[level_idx])
            flattened_stu_points.append(stu_points)
            flattened_strides.append(strides)
            tea_points = mlvl_tea_points[level_idx][teacher_indices[:, 1], teacher_indices[:, 0]]
            flattened_tea_points.append(tea_points)

        # Concatenate across all feature levels
        flatten_valid_stu_preds = (
            torch.cat([x[0] for x in flattened_stu_preds], dim=0),
            torch.cat([x[1] for x in flattened_stu_preds], dim=0),
            torch.cat([x[2] for x in flattened_stu_preds], dim=0),
        )

        flatten_valid_tea_preds = (
            torch.cat([x[0] for x in flattened_tea_preds], dim=0),
            torch.cat([x[1] for x in flattened_tea_preds], dim=0),
            torch.cat([x[2] for x in flattened_tea_preds], dim=0),
        )

        flatten_valid_stu_points = torch.cat(flattened_stu_points, dim=0)
        flatten_valid_tea_points = torch.cat(flattened_tea_points, dim=0)
        flatten_mlvl_strides = torch.cat(flattened_strides, dim=0)

        # Decode bounding boxes for student and teacher predictions
        flatten_valid_stu_bboxes = distance2obb(flatten_valid_stu_points, flatten_valid_stu_preds[1])
        flatten_valid_stu_bboxes = mintheta_obb(flatten_valid_stu_bboxes)
        flatten_valid_tea_bboxes = distance2obb(flatten_valid_tea_points, flatten_valid_tea_preds[1])
        flatten_valid_tea_bboxes = mintheta_obb(flatten_valid_tea_bboxes)
            
        with torch.no_grad():
            flatten_pseudo_obbox_targets, _ = self._transform_bbox(
                'obb', flatten_valid_tea_bboxes,
                trans_matrix
            )
            flatten_pseudo_obbox_targets = mintheta_obb(flatten_pseudo_obbox_targets)
            flatten_valid_bbox_pred_targets = obb2distance(flatten_valid_stu_points, 
                                                                flatten_pseudo_obbox_targets)
            if self.student.bbox_head.norm_on_bbox:
                flatten_valid_bbox_pred_targets[:, :4] = \
                    flatten_valid_bbox_pred_targets[:, :4] / flatten_mlvl_strides
                flatten_pseudo_obbox_targets = distance2obb(flatten_valid_stu_points, 
                                                        flatten_valid_bbox_pred_targets)
                
            flatten_valid_pred_targets = (flatten_valid_tea_preds[0], 
                                            flatten_valid_bbox_pred_targets,
                                            flatten_valid_tea_preds[2])

        return flatten_valid_stu_preds, flatten_valid_pred_targets, \
                flatten_valid_stu_bboxes, flatten_pseudo_obbox_targets, flatten_valid_tea_bboxes, \
                flatten_valid_stu_points, flatten_valid_tea_points, flatten_mlvl_strides

    def get_valid_preds_targets(self,
        batched_student_preds, batched_teacher_preds,
        valid_student_masks, teacher_mapping_indices,
        mlvl_stu_points, mlvl_tea_points, batched_trans_matrices
    ):
        """Process valid predictions and targets for a batch of images.

        Args:
            batched_student_preds (list[list[Tensor]]): Student predictions for each image in the batch.
            batched_teacher_preds (list[list[Tensor]]): Teacher predictions for each image in the batch.
            valid_student_masks (list[list[Tensor]]): Valid student masks for each image in the batch.
            teacher_mapping_indices (list[list[Tensor]]): Teacher mapping indices for each image in the batch.
            mlvl_stu_points (list[Tensor]): Student points [(H, W, 2), ...] 
                (same for all images in the batch).
            mlvl_tea_points (list[Tensor]): Teacher points [(H, W, 2), ...]
                (same for all images in the batch).

        Returns:
            tuple: Batch-level results as lists of flattened predictions and decoded bounding boxes.
        """
        batch_flatten_valid_stu_preds, batch_flatten_valid_pred_targets = [], []
        batch_flatten_valid_stu_bboxes, batch_flatten_pseudo_obbox_targets = [], []
        batch_flatten_valid_tea_bboxes, batch_flatten_valid_strides = [], []
        batch_flatten_valid_stu_points, batch_flatten_valid_tea_points = [], []

        for i in range(len(batched_student_preds[0])):
            res = self.get_valid_preds_targets_single(
                [pred[i] for pred in batched_student_preds], 
                [pred[i] for pred in batched_teacher_preds],
                valid_student_masks[i], teacher_mapping_indices[i],
                mlvl_stu_points, mlvl_tea_points, batched_trans_matrices[i]
            )
            batch_flatten_valid_stu_preds.append(res[0])
            batch_flatten_valid_pred_targets.append(res[1])
            batch_flatten_valid_stu_bboxes.append(res[2])
            batch_flatten_pseudo_obbox_targets.append(res[3])
            batch_flatten_valid_tea_bboxes.append(res[4])
            batch_flatten_valid_stu_points.append(res[5])
            batch_flatten_valid_tea_points.append(res[6])
            batch_flatten_valid_strides.append(res[7])
            

        return batch_flatten_valid_stu_preds, batch_flatten_valid_pred_targets, \
                batch_flatten_valid_stu_bboxes, batch_flatten_pseudo_obbox_targets, \
                batch_flatten_valid_tea_bboxes, \
                batch_flatten_valid_stu_points, batch_flatten_valid_tea_points, \
                batch_flatten_valid_strides

    def teacher_dets_to_masks(self, 
                              teacher_dets, 
                              image_shapes, 
                              trans_matrices,
                              to_normalize=True):
        """
        Convert teacher detection OBBs (oriented bounding boxes) into binary masks on the student images.

        Parameters:
        teacher_dets (list of torch.Tensor): 
            List of 2D tensors, each representing OBB detections for an image in the 
            teacher space with shape (N, 6), where N is the number of detections and each 
            OBB is represented as (x, y, w, h, theta, score).
        image_shapes (list of tuple or torch.Size): List of tuples (H, W) or torch.Size 
            representing the size of each student image in the batch.
        trans_matrices (list of torch.Tensor): List of transformation matrices to map 
            OBBs from teacher space to student space.
        to_normalize (bool): Whether to normalize the masks from image level to feature layer level.

        Returns:
        list of torch.Tensor: A list of boolean masks (one per image) of shape (H, W) or 
            (H // Stride, W // Stride) corresponding to the student image size or the featmap size.
        """

        def transform_and_convert_to_masks(dets, img_shape, trans_matrix):
            """
            Helper function to transform detections and convert them to mask for a single image.
            """
            # Step 1: Filter detections based on score threshold
            scores = dets[:, 5]  # Scores are in the last column
            valid_indices = scores > self.teacher_dets2mask_threshold
            filtered_dets = dets[valid_indices]
            # If need normalization, only consider the first layer of the feature maps
            stride = self.student.bbox_head.strides[0]
            shape_ = img_shape
            if to_normalize:
                shape_ = (img_shape[0] // stride, img_shape[1] // stride)

            if filtered_dets.size(0) == 0:
                # No valid detections, return a mask of all False
                H, W = shape_
                return torch.zeros((H, W), dtype=torch.bool).to(filtered_dets.device)

            # Step 2: Transform the OBBs (teacher to student space)
            transformed_obboxes, _ = self._transform_bbox('obb', filtered_dets[:, :5], trans_matrix)
            
            # Step 3: Convert transformed OBBs to polygons (N, 4, 2) using existing util function obb2poly
            polys = obb2poly(transformed_obboxes).reshape(-1, 4, 2)
            if to_normalize:
                polys /= stride

            # Step 4: Generate mask for the given image size
            mask = self.polygons2masks_single(polys, shape_)
            return mask

        # Apply the transformation, polygon conversion, and mask generation to each image in the batch
        masks = []
        for dets, img_shape, trans_matrix in zip(teacher_dets, image_shapes, trans_matrices):
            mask = transform_and_convert_to_masks(dets, img_shape, trans_matrix)
            masks.append(mask)

        return masks
    
    @staticmethod
    def polygons2masks_single(polygons: torch.Tensor, shape_: tuple):
        """
        Generate a 2D boolean mask tensor based on quadrilateral polygons using OpenCV.

        Parameters:
        polygons (torch.Tensor): Tensor of shape (N, 4, 2) representing 
            N polygons with 4 vertices in 2D space.
        shape_ (tuple): Tuple of (H, W) representing the height and width of the image or featmap.

        Returns:
        torch.Tensor: A boolean mask tensor of shape (H, W) where True values 
            represent the pixels inside any polygon.
        """
        H, W = shape_
        mask = np.zeros((H, W), dtype=np.uint8)  # Initialize a mask as a numpy array
        
        # Convert polygons to the format required by OpenCV
        for polygon in polygons:
            polygon_np = polygon.cpu().numpy().astype(np.int32)
            # Use cv2.fillPoly to fill the polygon in the mask
            cv2.fillPoly(mask, [polygon_np], 1)

        # Convert the mask to a PyTorch tensor and return it as a boolean tensor
        mask_tensor = torch.from_numpy(mask).bool().to(polygons.device)
        return mask_tensor

    def calculate_gc_ot_loss(self, 
                          batched_valid_stu_preds, 
                          batched_valid_pred_targets, 
                          valid_student_masks):
        """
        Calculate the optimal transport loss for the global consistency constraints 
        between student and teacher predictions.

        Parameters:
        batched_valid_stu_preds (list): List of student predictions.
        batched_valid_pred_targets (list): List of pseudo teacher predictions.
        valid_student_masks (list): List of valid student masks.

        Returns:
        torch.Tensor: The computed optimal transport loss.
        """
        loss_gc = batched_valid_pred_targets[0][0].new_tensor(0.)
        for i in range(len(batched_valid_pred_targets)):
            mask = valid_student_masks[i][0]
            assert mask.sum() == batched_valid_pred_targets[i][0].shape[0]
            if mask.sum() == 0:
                continue
            if self.ot_type in ['ot_loss_norm']:
                t_score_map = torch.softmax(batched_valid_pred_targets[i][0], dim=-1)
                s_score_map = torch.softmax(batched_valid_stu_preds[i][0], dim=-1)
            elif self.ot_type in ['ot_ang_loss_norm']:
                t_score_map = torch.abs(
                    batched_valid_pred_targets[i][1][:, -1, None]) / np.pi
                s_score_map = torch.abs(
                    batched_valid_stu_preds[i][1][:, -1, None]) / np.pi
            
            # Extract the max class scores within each image's slice
            t_score, t_score_cls = torch.max(t_score_map, dim=-1)
            s_score = s_score_map[torch.arange(t_score.shape[0], 
                                               device=t_score.device), t_score_cls]
            pts = mask.nonzero(as_tuple=False)[:, :2]
            loss_gc += self.gc_loss(
                s_score, t_score, pts, cost_type=self.cost_type, 
                clamp_ot=self.clamp_ot
            )
        return {'loss_gc': self.ot_weight * loss_gc}

    def visualize_dets_pts_test(self,
                       teacher_info,
                       student_info,
                       M_tea_stu,
                       M_stu_tea,
                       batched_valid_stu_points,
                       batched_valid_tea_points):
        
        dets_on_stu, _ = self._transform_bbox(
            'obb', teacher_info["det_bboxes"], M_tea_stu
        )
        visualize_images_with_points(
            student_info["img"], student_info["img_metas"],
            dets_on_stu, teacher_info["det_labels"],
            batched_valid_stu_points, 
            "student_pts_2", "data/DOTA/show_data_trans_stdt",
            score_thr=self.teacher_dets2mask_threshold
        )
        visualize_images_with_points(
            teacher_info["img"], teacher_info["img_metas"],
            teacher_info["det_bboxes"], teacher_info["det_labels"],
            batched_valid_tea_points, 
            "teacher_pts_2", "data/DOTA/show_data_trans_stdt",
            score_thr=self.teacher_dets2mask_threshold
        )
        tea_points_on_stu = []
        for tea_points, M in zip(batched_valid_tea_points, M_tea_stu):
            tea_points = torch.cat([tea_points, torch.ones_like(tea_points[:, 0:1])], dim=1)
            tea_points = torch.matmul(M, tea_points.t()).t()
            tea_points_on_stu.append(tea_points[:, :2] / tea_points[:, 2:3])
        visualize_images_with_points(
            student_info["img"], student_info["img_metas"],
            dets_on_stu, teacher_info["det_labels"],
            tea_points_on_stu, 
            "tea_pts_on_stu_2", "data/DOTA/show_data_trans_stdt",
            score_thr=self.teacher_dets2mask_threshold
        )
        stu_points_on_tea = []
        for stu_points, M in zip(batched_valid_stu_points, M_stu_tea):
            stu_points = torch.cat([stu_points, torch.ones_like(stu_points[:, 0:1])], dim=1)
            stu_points = torch.matmul(M, stu_points.t()).t()
            stu_points_on_tea.append(stu_points[:, :2] / stu_points[:, 2:3])
        visualize_images_with_points(
            teacher_info["img"], teacher_info["img_metas"],
            teacher_info["det_bboxes"], teacher_info["det_labels"],
            stu_points_on_tea, 
            "stu_pts_on_tea_2", "data/DOTA/show_data_trans_stdt",
            score_thr=self.teacher_dets2mask_threshold
        )