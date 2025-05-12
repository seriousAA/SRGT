from typing import Dict
from collections import OrderedDict

import numpy as np
import torch
import math
import torch.nn.functional as F
import torch.distributed as dist
import os
import random

from mmdet.models import BaseDetector, SingleStageDetector
from mmcv.runner.fp16_utils import force_fp32
from ssod.models.utils import Transform2D
from mmdet.core import bbox2type, multi_apply

class MultiStreamSTDetector(BaseDetector):
    def __init__(
        self, model: Dict[str, SingleStageDetector], train_cfg=None, test_cfg=None
    ):
        super(MultiStreamSTDetector, self).__init__()
        self.submodules = list(model.keys())
        for k, v in model.items():
            setattr(self, k, v)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.inference_on = self.test_cfg.get("inference_on", self.submodules[0])

    def model(self, **kwargs) -> SingleStageDetector:
        if "submodule" in kwargs:
            assert (
                kwargs["submodule"] in self.submodules
            ), "Detector does not contain submodule {}".format(kwargs["submodule"])
            model: SingleStageDetector = getattr(self, kwargs["submodule"])
        else:
            model: SingleStageDetector = getattr(self, self.inference_on)
        return model

    def freeze(self, model_ref: str):
        assert model_ref in self.submodules
        model = getattr(self, model_ref)
        model.eval()
        for param in model.parameters():
            param.requires_grad = False

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
        self.freeze("teacher") # It is a must, because the freezing op in the init func may
                                # get cancelled after the init due to the model loading of MMDetection.

    def forward_test(self, imgs, img_metas, **kwargs):

        return self.model(**kwargs).forward_test(imgs, img_metas, **kwargs)

    async def aforward_test(self, *, img, img_metas, **kwargs):
        return self.model(**kwargs).aforward_test(img, img_metas, **kwargs)

    def extract_feat(self, imgs):
        return self.model().extract_feat(imgs)

    async def aforward_test(self, *, img, img_metas, **kwargs):
        return self.model(**kwargs).aforward_test(img, img_metas, **kwargs)

    def aug_test(self, imgs, img_metas, **kwargs):
        return self.model(**kwargs).aug_test(imgs, img_metas, **kwargs)

    def simple_test(self, img, img_metas, **kwargs):
        return self.model(**kwargs).simple_test(img, img_metas, **kwargs)

    async def async_simple_test(self, img, img_metas, **kwargs):
        return self.model(**kwargs).async_simple_test(img, img_metas, **kwargs)

    def show_result(self, *args, **kwargs):
        self.model().CLASSES = self.CLASSES
        return self.model().show_result(*args, **kwargs)

    @force_fp32(apply_to=["bboxes", "trans_mat"])
    def _transform_bbox(self, bbox_type, bboxes, trans_mat, max_shape=None):
        return Transform2D.transform_bboxes(bbox_type, bboxes, trans_mat, max_shape)

    @force_fp32(apply_to=["a", "b"])
    def _get_trans_mat(self, a, b):
        return [bt @ at.inverse() for bt, at in zip(b, a)]

    @staticmethod
    def bbox2type_with_score(bboxes, to_type):
        return torch.cat([bbox2type(bboxes[..., :-1], to_type), bboxes[..., -1, None]], dim=-1)

    @staticmethod
    def map_norm_obbox_to_imgs(obboxes, points, strides):
        if isinstance(obboxes, list):
            assert len(obboxes) == len(points) and len(points) == len(strides)
            obboxes_ = []
            for obbox, point, stride in zip(obboxes, points, strides):
                obboxes_.append(MultiStreamSTDetector.map_norm_obbox_to_imgs(
                    obbox, point, stride
                ))
            return obboxes_
        assert obboxes.shape[0] == points.shape[0] and points.shape[0] == strides.shape[0]
        obboxes_ = obboxes.clone()
        obboxes_[:, :2] -= points
        obboxes_[:, :4] *= strides
        obboxes_[:, :2] += points
        return obboxes_

    def _parse_losses(self, losses):
        """Parse the raw outputs (losses) of the network.

        Args:
            losses (dict): Raw output of the network, which usually contain
                losses and other necessary infomation.

        Returns:
            tuple[Tensor, dict]: (loss, log_vars), loss is the loss tensor \
                which may be a weighted sum of all losses, log_vars contains \
                all the variables to be sent to the logger.
        """
        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            else:
                raise TypeError(
                    f'{loss_name} is not a tensor or list of tensors')

        loss = sum(_value for _key, _value in log_vars.items()
                   if 'loss' in _key)

        log_vars['loss'] = loss
        for loss_name, loss_value in log_vars.items():
            # reduce loss when distributed training
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            log_vars[loss_name] = loss_value.item()

        return loss, log_vars

    def train_step(self, data, optimizer):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.

        Args:
            data (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``, \
                ``num_samples``.

                - ``loss`` is a tensor for back propagation, which can be a
                  weighted sum of multiple losses.
                - ``log_vars`` contains all the variables to be sent to the
                  logger.
                - ``num_samples`` indicates the batch size (when the model is
                  DDP, it means the batch size on each GPU), which is used for
                  averaging the logs.
        """
        losses = self(**data)

        # warmup on unsupervised losses can stabilize the training
        if "unsup_start_iter" in os.environ:
            cur_iter = self.cur_iter
            unsup_start = int(os.environ["unsup_start_iter"])
            unsup_warmup = int(os.environ["unsup_warmup_iter"])

            if cur_iter < unsup_start + unsup_warmup:
                loss_weight = 0 if cur_iter < unsup_start \
                    else (cur_iter - unsup_start) / unsup_warmup
                for _key, _value in losses.items():
                    if _key.startswith('unsup') and '_num' not in _key and '_thr' not in _key and \
                                        '_rate' not in _key and 'loss' in _key:
                        if isinstance(_value, torch.Tensor):
                            losses[_key] = _value * loss_weight
                        elif isinstance(_value, list):
                            losses[_key] = [item * loss_weight for item in _value]

        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data['img_metas']))

        return outputs

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        if not any(["student" in key or "teacher" in key for key in state_dict.keys()]):
            keys = list(state_dict.keys())
            state_dict.update({"teacher." + k: state_dict[k] for k in keys})
            state_dict.update({"student." + k: state_dict[k] for k in keys})
            for k in keys:
                state_dict.pop(k)

        return super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
    
    def get_valid_mapping_masks_indices(self, 
                                        student_grid_points_list, 
                                        teacher_grid_points_list, 
                                        trans_matrices_list):
        """
        Match student grid points to teacher grid points for multiple feature map layers 
        across a batch of images.

        Parameters:
        - student_grid_points_list (list of torch.Tensor): 
            List of grid points on the student feature map layers across all images in the batch. 
            The list has the same length as the number of featmap layers.
            Each tensor in the list has the shape (H_s, W_s, 2).
        - teacher_grid_points_list (list of torch.Tensor): 
            List of grid points on the teacher feature map layers across all images in the batch. 
            The list has the same length as the number of featmap layers.
            Each tensor in the list has the shape (H_t, W_t, 2).
        - trans_matrices_list (list of torch.Tensor): 
            List of transformation matrices for each image in the batch.
            The length of this list is equal to the batch size. 
            Each matrix is used for all layers of the same image.

        Returns:
        - valid_student_masks (list[list[torch.Tensor]]): 
            A two-level list of boolean masks for student points.
            The outer list represents images in the batch, 
            and the inner list represents layers for each image.
        - teacher_mapping_indices (list[list[torch.Tensor]]): 
            A two-level list of mapping indices for matched teacher points. 
            The outer list represents images in the batch, 
            and the inner list represents layers for each image.
        """
        
        # Initialize the output lists for masks and mapping indices
        valid_student_masks = []
        teacher_mapping_indices = []
        assert len(student_grid_points_list) == len(teacher_grid_points_list)

        # Loop through each image in the batch
        for img_idx, trans_matrix in enumerate(trans_matrices_list):
            # For each image, apply multi_apply to process all feature map layers
            student_masks_per_image, teacher_indices_per_image = multi_apply(
                self.match_points_on_single_feat,
                student_grid_points_list,    # Student grid points list for all layers
                teacher_grid_points_list,    # Teacher grid points list for all layers
                # Strides for each feature map level
                self.teacher.bbox_head.strides[-len(student_grid_points_list):],  
                trans_matrix=trans_matrix    # Transformation matrix for this image
            )
            
            # Append the results for this image
            valid_student_masks.append(student_masks_per_image)
            teacher_mapping_indices.append(teacher_indices_per_image)
        
        return valid_student_masks, teacher_mapping_indices
    
    def match_points_on_single_feat(self, 
                                    student_grid_points, 
                                    teacher_grid_points,
                                    stride, 
                                    trans_matrix):
        """
        Match student grid points to teacher grid points based on a transformation matrix 
        for one single featmap layer on one image.

        Parameters:
        - student_grid_points (torch.Tensor): Grid points on the student feature map with shape (H_s, W_s, 2).
        - teacher_grid_points (torch.Tensor): Grid points on the teacher feature map with shape (H_t, W_t, 2).
        - trans_matrix (torch.Tensor): 3x3 transformation matrix to project student grid 
            points onto teacher space.
        - stride (int): Stride of the feature map.

        Returns:
        - student_mask_per_stu_lvl (torch.Tensor): Boolean mask of shape (H_s, W_s) indicating 
            if student points were matched.
        - mapping_indices_per_stu_lvl (torch.Tensor): Tensor of shape (H_s, W_s, 2) storing matched 
            teacher indices (if any).
        """
        H_s, W_s = student_grid_points.shape[:2]
        H_t, W_t = teacher_grid_points.shape[:2]
        device = student_grid_points.device
        
        # Step 1: Get the matching radius in original image space from teacher grid stride
        radius = stride / math.sqrt(2.0)
        
        # Step 2: Initialize output tensors
        student_mask_per_stu_lvl = torch.zeros((H_s, W_s), dtype=torch.bool, device=device)
        # -1 for unmatched points
        mapping_indices_per_stu_lvl = torch.full((H_s, W_s, 2), -1, dtype=torch.short, device=device)
        
        # Step 3: Convert student grid points to homogeneous coordinates for transformation
        # Add 1 for homogeneous coordinates
        student_grid_points_h = torch.cat([student_grid_points, 
                                        torch.ones((H_s, W_s, 1), device=device)], dim=2)
        
        # Step 4: Project student points to teacher space using the transformation matrix
        projected_student_points_h = torch.matmul(student_grid_points_h, trans_matrix.T)
        
        # Normalize projected points (divide by the third dimension to get 2D coordinates)
        projected_student_points = projected_student_points_h[..., :2] / projected_student_points_h[..., 2:]
        
        # Step 5: Filter out points that fall outside the teacher image bounds (original image size)
        valid_mask = (projected_student_points[..., 0] >= 0) & \
                    (projected_student_points[..., 0] < H_t * stride) & \
                    (projected_student_points[..., 1] >= 0) & \
                    (projected_student_points[..., 1] < W_t * stride)
        
        # Step 6: Flatten valid projected student points for easy distance calculation
        valid_projected_points = projected_student_points[valid_mask]  # Shape (N_valid, 2)
        valid_student_indices = valid_mask.nonzero(as_tuple=False)  # Get valid student indices (N_valid, 2)
        
        if valid_projected_points.size(0) == 0:
            # No valid points, return empty masks and indices
            return student_mask_per_stu_lvl, mapping_indices_per_stu_lvl
        
        # Step 7: Flatten teacher grid points for distance calculation
        teacher_grid_points_flat = teacher_grid_points.reshape(-1, 2)  # Shape (H_t * W_t, 2)
        
        # Step 8: Perform batched Euclidean distance calculations
        expanded_projected_points = valid_projected_points.unsqueeze(1)  # Shape (N_valid, 1, 2)
        expanded_teacher_points = teacher_grid_points_flat.unsqueeze(0)  # Shape (1, H_t * W_t, 2)
        
        # Compute pairwise Euclidean distances
        # Shape (N_valid, H_t * W_t)
        distances = torch.norm(expanded_projected_points - expanded_teacher_points, dim=2)  
        
        # Step 9: Find the closest teacher point within the radius for each student point
        # Boolean mask of points within the radius, Shape (N_valid, H_t * W_t)
        within_radius = distances < radius  
        
        if within_radius.any():
            closest_teacher_dists, closest_teacher_indices = \
                torch.min(distances + (~within_radius) * 1e6, dim=1)
            
            # Step 10: Get the corresponding 2D indices for the closest teacher grid points
            closest_teacher_grid_points = teacher_grid_points_flat[closest_teacher_indices]
            
            # Calculate the feature map indices for the closest teacher point
            teacher_grid_indices = \
                ((closest_teacher_grid_points - stride // 2) / stride).round().abs().short()
            
            # Step 11: Use a mask to filter out the valid matches
            valid_matches_mask = closest_teacher_dists < radius
            
            # Update student_mask_per_stu_lvl and mapping_indices_per_stu_lvl based on valid matches
            # Get the indices of valid matches
            matched_student_indices = valid_student_indices[valid_matches_mask]
            # Get the teacher grid indices of valid matches
            matched_teacher_indices = teacher_grid_indices[valid_matches_mask]
            
            # Update the mask for matched student points
            student_mask_per_stu_lvl[matched_student_indices[:, 0], \
                                    matched_student_indices[:, 1]] = True
            
            # Update the corresponding teacher indices for matched student points
            mapping_indices_per_stu_lvl[matched_student_indices[:, 0], \
                                    matched_student_indices[:, 1]] = matched_teacher_indices
        
        return student_mask_per_stu_lvl, mapping_indices_per_stu_lvl

    @staticmethod
    def reshape_batched_predictions(mlvl_batched_results, to_permute=False):
        """
            Reshape the list of batched predictions from [(B, H, W, C) or (B, C, H, W) for each feat layer] to 
            [[(H, W, C) for each feat layer] for each image in batch].

        Args:
            mlvl_batched_results (tuple): Batched predictions, each element is 
                a list of (B, H, W, C) or (B, C, H, W) shaped tensors.

        Returns:
            reshaped_batched_mlvl_results (list[list[Tensor]]): 
                Two-layer list of predictions for each image and each feature map level.
        """
        def reshape_preds_single(mlvl_batched_preds, to_permute):
            batch_size = mlvl_batched_preds[0].shape[0]
            num_levels = len(mlvl_batched_preds)
            reshaped_preds = [[] for _ in range(batch_size)]
            for level_idx in range(num_levels):
                for img_idx in range(batch_size):
                    if to_permute:
                        reshaped_preds[img_idx].append(
                            mlvl_batched_preds[level_idx][img_idx].permute(1, 2, 0))
                    else:
                        reshaped_preds[img_idx].append(
                            mlvl_batched_preds[level_idx][img_idx])
            return reshaped_preds
        
        if isinstance(mlvl_batched_results, (tuple, list)) and \
            isinstance(mlvl_batched_results[0], (tuple, list)):
            reshaped_batched_mlvl_results = [reshape_preds_single(preds, to_permute) 
                                           for preds in mlvl_batched_results]
        else:
            reshaped_batched_mlvl_results = reshape_preds_single(mlvl_batched_results)

        return reshaped_batched_mlvl_results
    
    @staticmethod
    def crop_images_to_pad_shape(img, img_metas):
        # Get the batch size
        B = img.size(0)
        
        # Extract the maximum height and width from img_metas
        max_h = max([img_metas[i]["pad_shape"][0] for i in range(B)])
        max_w = max([img_metas[i]["pad_shape"][1] for i in range(B)])
        
        # Crop the images in the batch
        cropped_imgs = img[:, :, :max_h, :max_w]
        
        return cropped_imgs
