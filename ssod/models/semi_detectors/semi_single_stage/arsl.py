import torch
import os
import torch.nn.functional as F
import copy

from mmdet.core import multi_apply, distance2obb, obb2distance, obb2poly, mintheta_obb
from mmdet.ops import obb_overlaps
from mmcv.ops import points_in_polygons
from mmdet.models import DETECTORS, build_detector
from mmdet.models.builder import build_loss
from mmdet.models.losses import accuracy

from ssod.utils.structure_utils import dict_split, weighted_loss
from ssod.utils import log_every_n
from ssod.models.utils import filter_invalid, process_visualization, visualize_images

from .multi_stream_st_detector import MultiStreamSTDetector


@DETECTORS.register_module()
class ARSLDetector(MultiStreamSTDetector):
    def __init__(self, model: dict, train_cfg=None, test_cfg=None):
        super(ARSLDetector, self).__init__(
            dict(teacher=build_detector(model), student=build_detector(model)),
            train_cfg=train_cfg,
            test_cfg=test_cfg,
        )
        self.freeze("teacher")
        if self.train_cfg is not None:
            self.unsup_weight = self.train_cfg.unsup_weight
            self.hard_neg_mining = self.train_cfg.get("hard_neg_mining", False)

            unsup_loss_cls_cfg = self.train_cfg.get("unsup_loss_cls", None)
            unsup_loss_bbbox_cfg = self.train_cfg.get("unsup_loss_bbbox", None)
            unsup_loss_iou_dist_cfg = self.train_cfg.get("unsup_loss_iou_dist", None)

            self.unsup_loss_cls = copy.deepcopy(self.student.bbox_head.loss_cls) \
                                if not unsup_loss_cls_cfg else build_loss(unsup_loss_cls_cfg)
            self.unsup_loss_iou_dist = copy.deepcopy(self.student.bbox_head.loss_iou_dist) \
                                if not unsup_loss_iou_dist_cfg else build_loss(unsup_loss_iou_dist_cfg)
            self.unsup_loss_bbox = copy.deepcopy(self.student.bbox_head.loss_bbox) \
                                if not unsup_loss_bbbox_cfg else build_loss(unsup_loss_bbbox_cfg)

            self.unsup_loss_cls.loss_weight = self.train_cfg.get("unsup_cls_weight", 1.0)
            self.unsup_loss_bbox.loss_weight = self.train_cfg.get("unsup_bbox_weight", 1.0)
            self.unsup_loss_iou_dist.loss_weight = self.train_cfg.get("unsup_iou_dist_weight", 0.5)

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
            teacher_info = self.extract_model_info(self.teacher, **teacher_info)
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
            teacher_info["iou_dist_preds"]
        )
        teacher_predictions = self.reshape_batched_predictions(teacher_predictions,
                                                               to_permute=True)
        student_predictions = (
            student_info["cls_scores"],
            student_info["bbox_preds"],
            student_info["theta_preds"],
            student_info["iou_dist_preds"]
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
                
        batched_valid_stu_preds, batched_valid_pred_targets, \
            batched_valid_stu_bboxes, batched_pseudo_obbox_targets, batched_valid_tea_bboxes, \
                batched_valid_stu_points, _, batched_valid_strides = \
                    self.get_valid_preds_targets(
                        student_predictions, teacher_predictions,
                        valid_student_masks, teacher_mapping_indices,
                        stu_mlvl_points, tea_mlvl_points, M_tea_stu
                    )

        # with torch.no_grad():
        #     visualize_images(
        #         student_info["img"], student_info["img_metas"],
        #         [t[::200] for t in self.map_norm_obbox_to_imgs(
        #             batched_valid_stu_bboxes, batched_valid_stu_points, batched_valid_strides)], 
        #         None,
        #         "student", "data/DOTA/show_data_trans_arsl"
        #     )
        #     visualize_images(
        #         student_info["img"], student_info["img_metas"],
        #         [t[::200] for t in self.map_norm_obbox_to_imgs(
        #             batched_pseudo_obbox_targets, batched_valid_stu_points, batched_valid_strides)], 
        #         None,
        #         "tea_on_student", "data/DOTA/show_data_trans_arsl"
        #     )
        #     visualize_images(
        #         teacher_info["img"], teacher_info["img_metas"],
        #         [t[::200] for t in batched_valid_tea_bboxes], 
        #         None,
        #         "teacher", "data/DOTA/show_data_trans_arsl"
        #     )

        cls_mask, loc_mask, cls_targets, loc_targets, iou_targets = \
            self.select_pseudo_targets(batched_valid_pred_targets, 
                                    batched_pseudo_obbox_targets,
                                    batched_valid_stu_points,
                                    batched_valid_strides)
        # Calculate training weights and average factors for loss normalization
        # Find positive indices for cls and loc tasks
        cls_pos_ind = torch.nonzero(cls_mask > 0).squeeze(dim=-1)
        loc_pos_ind = torch.nonzero(loc_mask > 0).squeeze(dim=-1)
        stu_cls_preds = torch.cat([pred[0] for pred in batched_valid_stu_preds], dim=0)
        stu_iou_dist_preds = torch.cat([pred[2] for pred 
                                                      in batched_valid_stu_preds], dim=0)
        stu_pred_decoded_bboxes = torch.cat(batched_valid_stu_bboxes, dim=0)
        
        # cls weight
        cls_avg_factor = torch.max(cls_targets[cls_pos_ind], dim=-1)[0].sum().item()
        cls_avg_factor = max(cls_avg_factor, 1.0)

        # loc weight
        loc_sample_weights = torch.max(cls_targets[loc_pos_ind], dim=-1)[0]
        loc_avg_factor = loc_sample_weights.sum().item()

        # with torch.no_grad():
        #     # assume that the batch size is 1 for simplicity
        #     valid_stu_points = torch.cat(batched_valid_stu_points, dim=0)
        #     valid_strides = torch.cat(batched_valid_strides, dim=0)
        #     visualize_images(
        #         student_info["img"], student_info["img_metas"],
        #         [self.map_norm_obbox_to_imgs(
        #             stu_pred_decoded_bboxes[loc_pos_ind], 
        #             valid_stu_points[loc_pos_ind], 
        #             valid_strides[loc_pos_ind])], 
        #         None,
        #         "pos_preds_on_student", "data/DOTA/show_data_trans_arsl"
        #     )
        #     visualize_images(
        #         student_info["img"], student_info["img_metas"],
        #         [self.map_norm_obbox_to_imgs(
        #             loc_targets[loc_pos_ind], 
        #             valid_stu_points[loc_pos_ind], 
        #             valid_strides[loc_pos_ind])], 
        #         None,
        #         "pos_targets_on_student", "data/DOTA/show_data_trans_arsl"
        #     )

        # iou weight
        iou_avg_factor = loc_pos_ind.shape[0]

        ### Compute Unsupervised Losses
        # Classification Loss (Quality Focal Loss with IoU guidance)
        if self.unsup_loss_cls.__class__.__name__ in ['SoftQFocalLossWithIoU']:
            loss_cls = self.unsup_loss_cls(
                stu_cls_preds,  # cls scores of student
                cls_targets,                 # cls targets (from teacher)
                # IoU predictions of student (guidance)
                implicit_iou=stu_iou_dist_preds.sigmoid().reshape(-1, 1), 
                avg_factor=cls_avg_factor
            )
        else:
            loss_cls = self.unsup_loss_cls(
                stu_cls_preds,  # cls scores of student
                cls_targets,                 # cls targets (from teacher)
                avg_factor=cls_avg_factor
            )

        loc_num_pos = loc_pos_ind.shape[0]
        # IoU-Distance Loss (Binary Cross Entropy for IoU quality)
        pos_stu_iou = stu_iou_dist_preds[loc_pos_ind]
        pos_iou_targets = iou_targets[loc_pos_ind]
        pos_stu_bboxes = stu_pred_decoded_bboxes[loc_pos_ind]
        pos_bbox_targets = loc_targets[loc_pos_ind]
        
        if loc_num_pos > 0:
            loss_iou_dist = self.unsup_loss_iou_dist(
                pos_stu_iou, pos_iou_targets,
                avg_factor=iou_avg_factor
            )

            # Box Loss (IoU Loss for Bounding Boxes)
            loss_bbox = self.unsup_loss_bbox(
                pos_stu_bboxes, pos_bbox_targets,
                weight=loc_sample_weights,
                avg_factor=loc_avg_factor
            )
        else:
            loss_iou_dist = pos_stu_iou.sum()
            loss_bbox = pos_stu_bboxes.sum()

        # Aggregate all losses into a single dictionary
        unsup_losses = {
            "loss_cls": loss_cls,
            "loss_bbox": loss_bbox,
            "loss_iou_dist": loss_iou_dist,
            "pos_loc_num": torch.tensor(loc_num_pos, dtype=torch.float).to(loss_cls.device) \
                / len(batched_valid_stu_preds)
        }

        return unsup_losses

    def hard_neg_samples_mining(self,
                        cls_score,
                        loc_obboxes,
                        quality,
                        pos_ind,
                        hard_neg_ind,
                        points,
                        strides,
                        loc_mask,
                        loc_targets,
                        iou_thresh=0.6):
        """Hard negative mining for ARSL OBB detection.

        Args:
            cls_score (Tensor): Classification scores (H, W, C).
            loc_obboxes (Tensor): Localization decoded bboxes (H, W, 5) for OBB.
            quality (Tensor): IoU-Dist/quality predictions (H, W, 1).
            pos_ind (Tensor): Indices of positive samples.
            hard_neg_ind (Tensor): Indices of hard negatives.
            points (Tensor): Points on feature maps (N, 2).
            loc_mask (Tensor): Localization mask.
            loc_targets (Tensor): Localization targets.
            iou_thresh (float): IoU threshold for considering a hard negative as a potential positive.

        Returns:
            tuple: Updated loc_mask and loc_targets after hard negative mining.
        """
        # Compute classification scores: cls * iou
        cls_vals = torch.sigmoid(cls_score) * torch.sigmoid(quality)
        max_vals, class_ind = torch.max(cls_vals, dim=-1)
        hard_neg_obboxes = loc_obboxes[hard_neg_ind]
        pos_obboxes = loc_obboxes[pos_ind]

        # Compute IoU between positive and hard negative OBBs
        hard_neg_pos_iou = obb_overlaps(hard_neg_obboxes, pos_obboxes).squeeze(-1)

        # Select potential positives from hard negatives
        # Stride(or Scale) flag (the stride ratios within 2.)
        stride_ratios = strides[hard_neg_ind].reshape([-1])[:, None]/ \
                        strides[pos_ind].reshape([-1])[None, :]
        stride_flag = (stride_ratios>=0.5) & (stride_ratios<=2.0)
        
        # IoU flag (consider IoU above a threshold)
        iou_flag = (hard_neg_pos_iou >= iou_thresh)

        # Class flag (same class between hard negative and positive samples)
        pos_classes = class_ind[pos_ind]
        hard_neg_classes = class_ind[hard_neg_ind]
        class_flag = (pos_classes.unsqueeze(0) == hard_neg_classes.unsqueeze(1))

        # Inside-OBB flag: Ensure that the hard negative point lies within the positive OBB
        hard_neg_points = points[hard_neg_ind]
        # Convert pos_obboxes to polygons
        pos_polygons = obb2poly(pos_obboxes)  # Shape: (N_pos, 8)
        inside_flag = points_in_polygons(hard_neg_points, pos_polygons) > 0  # Updated to handle OBBs

        # Combined valid flag for hard negatives
        valid_flag = stride_flag & iou_flag & class_flag & inside_flag

        # Zero-out invalid IoUs
        hard_neg_pos_iou = torch.where(valid_flag, hard_neg_pos_iou, torch.zeros_like(hard_neg_pos_iou))

        # Select potential positive indices
        pos_hard_neg_max_iou = hard_neg_pos_iou.max(dim=-1)[0]
        potential_pos_ind = (pos_hard_neg_max_iou > 0).nonzero(as_tuple=True)[0]
        if potential_pos_ind.numel() == 0:
            return None

        # Prepare data: potential points, strides, and valid flags
        potential_valid_flag = valid_flag[potential_pos_ind]
        potential_pos_ind = hard_neg_ind[potential_pos_ind]
        num_potential_pos = potential_pos_ind.shape[0]
        
        # Get classification scores and bounding boxes of matching positives
        pos_cls = max_vals[pos_ind]
        
        # Expand positive bounding boxes to match the number of potential positives
        expand_pos_bbox = pos_obboxes.unsqueeze(0).expand(num_potential_pos, pos_obboxes.size(0), 5)
        
        # Expand positive class scores
        expand_pos_cls = pos_cls.unsqueeze(0).expand(num_potential_pos, pos_cls.size(0))
        invalid_cls = torch.zeros_like(expand_pos_cls)
        
        # Apply valid flag to class scores
        expand_pos_cls = torch.where(potential_valid_flag, expand_pos_cls, invalid_cls)
        expand_pos_cls = expand_pos_cls.unsqueeze(-1)

        # Aggregate boxes based on class scores
        agg_bbox = (expand_pos_bbox * expand_pos_cls).sum(dim=1) / expand_pos_cls.sum(dim=1)
        
        # Assign the aggregated bounding boxes directly to the localization targets (no encoding)
        loc_targets[potential_pos_ind] = agg_bbox
        loc_mask[potential_pos_ind] = 1.0

        return loc_mask, loc_targets

    def select_pseudo_targets_per_img(self, flatten_valid_target_preds, flatten_valid_target_bboxes, 
                                   flatten_valid_stu_points, flatten_valid_strides):
        # Extract teacher predictions
        tea_cls, _, tea_iou = flatten_valid_target_preds

        # Prepare data: Compute scores from teacher's predictions
        tea_cls_scores = torch.sigmoid(tea_cls) * torch.sigmoid(tea_iou)
        class_ind = torch.argmax(tea_cls_scores, dim=-1)
        max_vals = torch.max(tea_cls_scores, dim=-1).values
        cls_mask = torch.zeros_like(max_vals)  # Class mask

        num_pos, num_hard_neg = 0, 0

        # Mean-std selection for positive and hard negative samples
        candidate_ind = torch.nonzero(max_vals >= 0.1).squeeze(dim=-1)
        num_candidate = candidate_ind.shape[0]

        if num_candidate > 0:
            candidate_score = max_vals[candidate_ind]
            candidate_score_mean = candidate_score.mean()
            candidate_score_std = candidate_score.std()
            pos_thresh = torch.clamp(candidate_score_mean + candidate_score_std, max=0.4)

            # Select positive samples
            pos_ind = torch.nonzero(max_vals >= pos_thresh).squeeze(dim=-1)
            num_pos = pos_ind.shape[0]

            # Select hard negatives as potential positives
            hard_neg_ind = torch.nonzero((max_vals >= 0.1) & (max_vals < pos_thresh)).squeeze(dim=-1)
            num_hard_neg = hard_neg_ind.shape[0]

        if num_pos == 0:
            num_pos = min(10, len(max_vals))
            _, pos_ind = torch.topk(max_vals, k=num_pos)

        cls_mask[pos_ind] = 1.0

        # Generate targets
        pos_class_ind = class_ind[pos_ind]
        cls_targets = torch.zeros_like(tea_cls)
        cls_targets[pos_ind, pos_class_ind] = tea_cls_scores[pos_ind, pos_class_ind]

        if num_hard_neg > 0:
            cls_targets[hard_neg_ind] = tea_cls_scores[hard_neg_ind]

        loc_targets = torch.zeros_like(flatten_valid_target_bboxes)
        loc_targets[pos_ind] = flatten_valid_target_bboxes[pos_ind]

        iou_targets = torch.zeros(tea_iou.shape[0], dtype=tea_iou.dtype, device=tea_iou.device)
        iou_targets[pos_ind] = torch.sigmoid(tea_iou.squeeze(-1)[pos_ind])

        loc_mask = cls_mask.clone()

        if num_hard_neg > 0 and self.hard_neg_mining:
            results = self.hard_neg_samples_mining(tea_cls, flatten_valid_target_bboxes, tea_iou, pos_ind, 
                                           hard_neg_ind, flatten_valid_stu_points, flatten_valid_strides,
                                           loc_mask, loc_targets)
            if results is not None:
                loc_mask, loc_targets = results
                loc_pos_ind = torch.nonzero(loc_mask > 0.0).squeeze(dim=-1)
                iou_targets[loc_pos_ind] = torch.sigmoid(tea_iou.squeeze(-1)[loc_pos_ind])

        return cls_mask, loc_mask, cls_targets, loc_targets, iou_targets

    def select_pseudo_targets(self, batched_valid_pred_targets,
                        batched_pseudo_obbox_targets,
                        batched_valid_stu_points,
                        batched_valid_strides):
        """
        Generate pseudo targets for each image in the batch.
        """
        # Use multi_apply to handle the per-image pseudo target generation
        cls_mask_list, loc_mask_list, cls_targets_list, \
            loc_targets_list, iou_targets_list = multi_apply(
                self.select_pseudo_targets_per_img, 
                batched_valid_pred_targets, 
                batched_pseudo_obbox_targets, 
                batched_valid_stu_points,
                batched_valid_strides
            )
        
        # Concatenate the results across the batch
        cls_mask = torch.cat(cls_mask_list, dim=0)
        loc_mask = torch.cat(loc_mask_list, dim=0)
        cls_targets = torch.cat(cls_targets_list, dim=0)
        loc_targets = torch.cat(loc_targets_list, dim=0)
        iou_targets = torch.cat(iou_targets_list, dim=0)
        
        return cls_mask, loc_mask, cls_targets, loc_targets, iou_targets

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

        # Obtain classification scores, bbox predictions, theta predictions, and IoU predictions directly.
        # Notice here that the tea_bbox_preds will be denormalized by the strides back to the image space
        # due to the testing mode of teacher model, while the student model will not do this due to its
        # training mode.
        cls_scores, bbox_preds, theta_preds, iou_dist_preds = model.bbox_head(feat)
        model_info["cls_scores"] = list(cls_scores)
        model_info["bbox_preds"] = list(bbox_preds)
        model_info["theta_preds"] = list(theta_preds)
        model_info["iou_dist_preds"] = list(iou_dist_preds)

        model_info["img_metas"] = img_metas
        model_info["transform_matrix"] = [
            torch.from_numpy(meta["transform_matrix"]).float().to(feat[0][0].device).requires_grad_(False)
            for meta in img_metas
        ]

        # Include ground truth data if provided (for supervised loss calculation)
        if "gt_obboxes" in kwargs:
            model_info["gt_obboxes"] = kwargs["gt_obboxes"]
            model_info["gt_labels"] = kwargs["gt_labels"]

        return model_info
    
    def get_valid_preds_targets_single(self,
        single_mlvl_student_preds, single_mlvl_teacher_preds,
        single_valid_student_masks, single_teacher_mapping_indices,
        mlvl_stu_points, mlvl_tea_points, trans_matrix
    ):
        """Process valid predictions and targets for a single image.

        Args:
            single_mlvl_student_preds (tuple): Student predictions 
            (cls_scores, bbox_preds, theta_preds, iou_dist_preds).
            single_mlvl_teacher_preds (tuple): Teacher predictions 
            (cls_scores, bbox_preds, theta_preds, iou_dist_preds).
            single_valid_student_masks (list[Tensor]): Valid student masks for 
            each feature level [(H, W), ...].
            single_teacher_mapping_indices (list[Tensor]): Teacher mapping indices 
            for valid student points [(H, W, 2), ...].
            mlvl_stu_points (list[Tensor]): Student points [(H, W, 2), ...] 
            (same for all images in the batch).

        Returns:
            tuple: Flattened valid student predictions, teacher predictions, and decoded bounding boxes.
                - flatten_valid_stu_preds: Tuple of flattened student predictions 
                    (cls_scores, bbox_preds, theta_preds, iou_dist_preds).
                - flatten_valid_stu_bboxes: Decoded student bounding boxes (N, 5) on student input images.
                - flatten_valid_tea_bboxes: Decoded teacher bounding boxes (N, 5) on teacher input images.
                - flatten_valid_stu_points: Flattened student points (N, 2) on student input images.
                - flatten_valid_tea_points: Flattened teacher points (N, 2) on teacher input images.
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
            stu_iou_dist_pred = single_mlvl_student_preds[3][level_idx][mask]

            # Combine the student predictions
            flattened_stu_preds.append((stu_cls_pred, torch.cat([stu_bbox_pred, 
                                            stu_theta_pred], dim=1), stu_iou_dist_pred))

            # Get corresponding teacher indices and map teacher predictions
            teacher_indices = single_teacher_mapping_indices[level_idx][mask].long()
            tea_cls_pred = single_mlvl_teacher_preds[0][level_idx][teacher_indices[:, 1], 
                                                                      teacher_indices[:, 0]]
            tea_bbox_pred = single_mlvl_teacher_preds[1][level_idx][teacher_indices[:, 1], 
                                                                       teacher_indices[:, 0]]
            tea_theta_pred = single_mlvl_teacher_preds[2][level_idx][teacher_indices[:, 1], 
                                                                        teacher_indices[:, 0]]
            tea_iou_dist_pred = single_mlvl_teacher_preds[3][level_idx][teacher_indices[:, 1], 
                                                                           teacher_indices[:, 0]]

            # Combine the teacher predictions
            flattened_tea_preds.append((tea_cls_pred, torch.cat([tea_bbox_pred, 
                                        tea_theta_pred], dim=1), tea_iou_dist_pred))

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
        flatten_valid_tea_bboxes = distance2obb(flatten_valid_tea_points, flatten_valid_tea_preds[1])
        flatten_valid_tea_bboxes = mintheta_obb(flatten_valid_tea_bboxes)
            
        with torch.no_grad():
            flatten_pseudo_obbox_targets, _ = self._transform_bbox(
                'obb', flatten_valid_tea_bboxes,
                trans_matrix
            )
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
