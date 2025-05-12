import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Scale, normal_init

from mmdet.core import (distance2obb, force_fp32, multi_apply, multiclass_arb_nms,
                        mintheta_obb)
from mmdet.models import HEADS, build_loss, OBBAnchorFreeHead
from mmdet.ops import obb_overlaps

INF = 1e8


@HEADS.register_module()
class OBBFCOSHeadWithIoU(OBBAnchorFreeHead):
    """Anchor-free head for FCOS with IoU-Distance prediction replacing centerness.

    This head is a modified version of the original FCOS head where the 
    centerness branch is replaced by an IoU prediction branch. The IoU-Distance 
    prediction directly measures the quality of the predicted bounding boxes 
    relative to the ground truth, providing a better metric for suppressing 
    low-quality predictions compared to centerness.

    This implementation is based on the Ambiguity-Resistant Semi-Supervised 
    Learning (ARSL) method for dense object detection, as described in the 
    paper available at: https://arxiv.org/abs/2303.14960.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        strides (list[int] | list[tuple[int, int]]): Strides of points
            in multiple feature levels. Default: (4, 8, 16, 32, 64).
        regress_ranges (tuple[tuple[int, int]]): Regress range of multiple
            level points.
        center_sampling (bool): If true, use center sampling. Default: False.
        center_sample_radius (float): Radius of center sampling. Default: 1.5.
        norm_on_bbox (bool): If true, normalize the regression targets
            with FPN strides. Default: False.
        iou_on_reg (bool): If true, position IoU prediction on the
            regression branch. This setting can be used to align the 
            regression and IoU prediction tasks more closely. Default: False.
        conv_bias (bool | str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias of conv will be set as True if `norm_cfg` is None, otherwise
            False. Default: "auto".
        loss_cls (dict): Config of classification loss.
        loss_bbox (dict): Config of localization loss.
        loss_iou_dist (dict): Config of IoU-Distance loss (to replace the original centerness loss).
        norm_cfg (dict): Dictionary to construct and configure the normalization layer.
            Default: norm_cfg=dict(type='GN', num_groups=32, requires_grad=True).
    """  # noqa: E501

    def __init__(self,
                 num_classes,
                 in_channels,
                 regress_ranges=((-1, 64), (64, 128), (128, 256), (256, 512),
                                 (512, INF)),
                 center_sampling=False,
                 center_sample_radius=1.5,
                 norm_on_bbox=False,
                 iou_on_reg=False,
                 scale_theta=True,
                 loss_cls=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0),
                 loss_bbox=dict(type='PolyIoULoss', loss_weight=1.0),
                 loss_iou_dist=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0),
                 norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
                 **kwargs):
        self.regress_ranges = regress_ranges
        self.center_sampling = center_sampling
        self.center_sample_radius = center_sample_radius
        self.norm_on_bbox = norm_on_bbox
        self.iou_on_reg = iou_on_reg
        self.scale_theta = scale_theta
        super().__init__(
            num_classes,
            in_channels,
            bbox_type='obb',
            reg_dim=4,
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            norm_cfg=norm_cfg,
            **kwargs)
        self.loss_iou_dist = build_loss(loss_iou_dist)

    def _init_layers(self):
        """Initialize layers of the head."""
        super()._init_layers()
        self.conv_iou_dist = nn.Conv2d(self.feat_channels, 1, 3, padding=1)
        self.conv_theta = nn.Conv2d(self.feat_channels, 1, 3, padding=1)
        self.scales = nn.ModuleList([Scale(1.0) for _ in self.strides])
        if self.scale_theta:
            self.scale_t = Scale(1.0)

    def init_weights(self):
        """Initialize weights of the head."""
        super().init_weights()
        normal_init(self.conv_iou_dist, std=0.01)
        normal_init(self.conv_theta, std=0.01)

    def forward(self, feats):
        """Forward features from the upstream network.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple:
                cls_scores (list[Tensor]): Box scores for each scale level,
                    each is a 4D-tensor, the channel number is
                    num_points * num_classes.
                bbox_preds (list[Tensor]): Box energies / deltas for each scale
                    level, each is a 4D-tensor, the channel number is
                    num_points * 4.
                iou_dist_preds (list[Tensor]): IoU values for each scale level,
                    each is a 4D-tensor, the channel number is num_points * 1.
        """
        return multi_apply(self.forward_single, feats, self.scales,
                           self.strides)

    def forward_single(self, x, scale, stride):
        """Forward features of a single scale levle.

        Args:
            x (Tensor): FPN feature maps of the specified stride.
            scale (:obj: `mmcv.cnn.Scale`): Learnable scale module to resize
                the bbox prediction.
            stride (int): The corresponding stride for feature maps, only
                used to normalize the bbox prediction when self.norm_on_bbox
                is True.

        Returns:
            tuple: scores for each class, bbox predictions and iou
                predictions of input feature maps.
        """
        cls_score, bbox_pred, cls_feat, reg_feat = super().forward_single(x)
        if self.iou_on_reg:
            iou = self.conv_iou_dist(reg_feat)
        else:
            iou = self.conv_iou_dist(cls_feat)
        # scale the bbox_pred of different level
        # float to avoid overflow when enabling FP16
        bbox_pred = scale(bbox_pred).float()
        if self.norm_on_bbox:
            bbox_pred = F.relu(bbox_pred)
            if not self.training:
                bbox_pred *= stride
        else:
            bbox_pred = bbox_pred.exp()
        theta_pred = self.conv_theta(reg_feat)
        if self.scale_theta:
            theta_pred = self.scale_t(theta_pred).float()
        return cls_score, bbox_pred, theta_pred, iou

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'theta_preds', 'iou_dist_preds'))
    def loss(self,
             cls_scores,
             bbox_preds,
             theta_preds,
             iou_dist_preds,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):
        """Compute loss of the head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level,
                each is a 4D-tensor, the channel number is
                num_points * num_classes.
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level, each is a 4D-tensor, the channel number is
                num_points * 4.
            iou_dist_preds (list[Tensor]): IoU values for each scale level, each
                is a 4D-tensor, the channel number is num_points * 1.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert len(cls_scores) == len(bbox_preds) == len(iou_dist_preds)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        all_level_points = self.get_points(featmap_sizes, bbox_preds[0].dtype,
                                           bbox_preds[0].device)
        labels, bbox_targets = self.get_targets(all_level_points, gt_bboxes,
                                                gt_labels)

        num_imgs = cls_scores[0].size(0)
        # flatten cls_scores, bbox_preds and iou
        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
            for cls_score in cls_scores
        ]
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
            for bbox_pred in bbox_preds
        ]
        flatten_theta_preds = [
            theta_pred.permute(0, 2, 3, 1).reshape(-1, 1)
            for theta_pred in theta_preds
        ]
        flatten_iou_dist_preds = [
            iou.permute(0, 2, 3, 1).reshape(-1)
            for iou in iou_dist_preds
        ]
        flatten_cls_scores = torch.cat(flatten_cls_scores)
        flatten_bbox_preds = torch.cat(flatten_bbox_preds)
        flatten_theta_preds = torch.cat(flatten_theta_preds)
        flatten_iou_dist_preds = torch.cat(flatten_iou_dist_preds)
        flatten_labels = torch.cat(labels)
        flatten_bbox_targets = torch.cat(bbox_targets)
        # repeat points to align with bbox_preds
        flatten_points = torch.cat(
            [points.repeat(num_imgs, 1) for points in all_level_points])

        # cat bbox_preds and theta_preds to obb bbox_preds
        flatten_bbox_preds = torch.cat(
            [flatten_bbox_preds, flatten_theta_preds], dim=1)

        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        bg_class_ind = self.num_classes
        pos_inds = ((flatten_labels >= 0)
                    & (flatten_labels < bg_class_ind)).nonzero().reshape(-1)
        num_pos = len(pos_inds)

        pos_bbox_preds = flatten_bbox_preds[pos_inds]
        pos_iou_dist_preds = flatten_iou_dist_preds[pos_inds]
        if num_pos > 0:
            pos_bbox_targets = flatten_bbox_targets[pos_inds]
            pos_points = flatten_points[pos_inds]
            pos_decoded_bbox_preds = distance2obb(pos_points, pos_bbox_preds)
            pos_decoded_target_preds = distance2obb(pos_points, pos_bbox_targets)
            pos_ious = obb_overlaps(pos_decoded_bbox_preds, pos_decoded_target_preds, 
                                                 is_aligned=True).squeeze(-1)
        if self.loss_cls.__class__.__name__ in ['SoftQFocalLossWithIoU']:
            flatten_cls_targets = flatten_cls_scores.new_zeros(
                flatten_cls_scores.size())
            pos_labels = flatten_labels[pos_inds].squeeze(-1)
            if num_pos > 0:
                flatten_cls_targets[pos_inds, pos_labels] = pos_ious
            loss_cls = self.loss_cls(
                flatten_cls_scores, flatten_cls_targets,
                implicit_iou=flatten_iou_dist_preds.sigmoid().reshape(-1, 1),
                avg_factor=max(num_pos, 1))  # avoid num_pos is 0
        else:
            loss_cls = self.loss_cls(
                flatten_cls_scores, flatten_labels,
                avg_factor=max(num_pos, 1))  # avoid num_pos is 0

        if num_pos > 0:
            pos_centerness = self.centerness_target(pos_bbox_targets)
            # iou weighted iou loss
            loss_bbox = self.loss_bbox(
                pos_decoded_bbox_preds,
                pos_decoded_target_preds,
                weight=pos_centerness,
                avg_factor=pos_centerness.sum())
            loss_iou_dist = self.loss_iou_dist(pos_iou_dist_preds,
                                                   pos_ious)
        else:
            loss_bbox = pos_bbox_preds.sum()
            loss_iou_dist = pos_iou_dist_preds.sum()

        return dict(
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            loss_iou_dist=loss_iou_dist)

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'theta_preds', 'iou_dist_preds'))
    def get_bboxes(self,
                   cls_scores,
                   bbox_preds,
                   theta_preds,
                   iou_dist_preds,
                   img_metas,
                   cfg=None,
                   rescale=None):
        """ Transform network output for a batch into bbox predictions.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_points * num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_points * 4, H, W)
            iou_dist_preds (list[Tensor]): iou for each scale level with
                shape (N, num_points * 1, H, W)
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used
            rescale (bool): If True, return boxes in original image space

        Returns:
             list[tuple[Tensor, Tensor]]: Each item in result_list is a 2-tuple.
                The first item is an (n, 6) tensor, where the first 5 columns
                represent the bounding box parameters (center_x, center_y, width, height, angle),
                and the 6th column is a score between 0 and 1. The second item is a
                (n,) tensor where each item is the predicted class label of the
                corresponding box.
        """
        assert len(cls_scores) == len(bbox_preds)
        num_levels = len(cls_scores)

        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        mlvl_points = self.get_points(featmap_sizes, bbox_preds[0].dtype,
                                      bbox_preds[0].device)
        result_list = []
        for img_id in range(len(img_metas)):
            cls_score_list = [
                cls_scores[i][img_id].detach() for i in range(num_levels)
            ]
            bbox_pred_list = [
                bbox_preds[i][img_id].detach() for i in range(num_levels)
            ]
            theta_pred_list = [
                theta_preds[i][img_id].detach() for i in range(num_levels)
            ]
            iou_pred_list = [
                iou_dist_preds[i][img_id].detach() for i in range(num_levels)
            ]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            det_bboxes = self._get_bboxes_single(cls_score_list,
                                                 bbox_pred_list,
                                                 theta_pred_list,
                                                 iou_pred_list,
                                                 mlvl_points, img_shape,
                                                 scale_factor, cfg, rescale)
            result_list.append(det_bboxes)
        return result_list

    def _get_bboxes_single(self,
                           cls_scores,
                           bbox_preds,
                           theta_preds,
                           iou_dist_preds,
                           mlvl_points,
                           img_shape,
                           scale_factor,
                           cfg,
                           rescale=False):
        """Transform outputs for a single batch item into bbox predictions.

        Args:
            cls_scores (list[Tensor]): Box scores for a single scale level
                Has shape (num_points * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for a single scale
                level with shape (num_points * 4, H, W).
            iou_dist_preds (list[Tensor]): iou for a single scale level
                with shape (num_points * 4, H, W).
            mlvl_points (list[Tensor]): Box reference for a single scale level
                with shape (num_total_points, 4).
            img_shape (tuple[int]): Shape of the input image,
                (height, width, 3).
            scale_factor (ndarray): Scale factor of the image arrange as
                (w_scale, h_scale, w_scale, h_scale).
            cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.

        Returns:
            Tensor: Labeled boxes in shape (n, 6), where the first 5 columns
                represent the bounding box parameters (center_x, center_y, width, height, angle),
                and the 6th column is a score between 0 and 1.
        """
        cfg = self.test_cfg if cfg is None else cfg
        assert len(cls_scores) == len(bbox_preds) == len(mlvl_points)
        mlvl_bboxes = []
        mlvl_scores = []
        mlvl_iou_dists = []
        for cls_score, bbox_pred, theta_pred, iou, points in zip(
                cls_scores, bbox_preds, theta_preds, iou_dist_preds, mlvl_points):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            scores = cls_score.permute(1, 2, 0).reshape(
                -1, self.cls_out_channels).sigmoid()
            iou = iou.permute(1, 2, 0).reshape(-1).sigmoid()

            theta_pred = theta_pred.permute(1, 2, 0).reshape(-1, 1)
            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            bbox_pred = torch.cat([bbox_pred, theta_pred], dim=1)
            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                max_scores, _ = (scores * iou[:, None]).max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                points = points[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                scores = scores[topk_inds, :]
                iou = iou[topk_inds]
            bboxes = distance2obb(points, bbox_pred, max_shape=img_shape)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
            mlvl_iou_dists.append(iou)
        mlvl_bboxes = torch.cat(mlvl_bboxes)
        if rescale:
            scale_factor = mlvl_bboxes.new_tensor(scale_factor)
            mlvl_bboxes[..., :4] = mlvl_bboxes[..., :4] / scale_factor
        mlvl_scores = torch.cat(mlvl_scores)
        padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
        # remind that we set FG labels to [0, num_class-1] since mmdet v2.0
        # BG cat_id: num_class
        mlvl_scores = torch.cat([mlvl_scores, padding], dim=1)
        mlvl_iou_dists = torch.cat(mlvl_iou_dists)
        det_bboxes, det_labels = multiclass_arb_nms(
            mlvl_bboxes,
            mlvl_scores,
            cfg.score_thr,
            cfg.nms,
            cfg.max_per_img,
            score_factors=mlvl_iou_dists,
            bbox_type='obb')
        return det_bboxes, det_labels

    def _get_points_single(self,
                           featmap_size,
                           stride,
                           dtype,
                           device,
                           no_flatten=False):
        """Get points according to feature map sizes."""
        y, x = super()._get_points_single(featmap_size, stride, dtype, device)
        if not no_flatten:
            y, x = y.reshape(-1), x.reshape(-1)
        points = torch.stack((x * stride, y * stride), dim=-1) + stride // 2
        return points

    def get_targets(self, points, gt_bboxes_list, gt_labels_list):
        """Compute regression, classification and iou-distance targets for points
            in multiple images.

        Args:
            points (list[Tensor]): Points of each fpn level, each has shape
                (num_points, 2).
            gt_bboxes_list (list[Tensor]): Ground truth bboxes of each image,
                each has shape (num_gt, 4).
            gt_labels_list (list[Tensor]): Ground truth labels of each box,
                each has shape (num_gt,).

        Returns:
            tuple:
                concat_lvl_labels (list[Tensor]): Labels of each level.
                concat_lvl_bbox_targets (list[Tensor]): BBox targets of each
                    level.
        """
        assert len(points) == len(self.regress_ranges)
        num_levels = len(points)
        # expand regress ranges to align with points
        expanded_regress_ranges = [
            points[i].new_tensor(self.regress_ranges[i])[None].expand_as(
                points[i]) for i in range(num_levels)
        ]
        # concat all levels points and regress ranges
        concat_regress_ranges = torch.cat(expanded_regress_ranges, dim=0)
        concat_points = torch.cat(points, dim=0)

        # the number of points per img, per lvl
        num_points = [center.size(0) for center in points]

        # get labels and bbox_targets of each image
        labels_list, bbox_targets_list = multi_apply(
            self._get_target_single,
            gt_bboxes_list,
            gt_labels_list,
            points=concat_points,
            regress_ranges=concat_regress_ranges,
            num_points_per_lvl=num_points)

        # split to per img, per level
        labels_list = [labels.split(num_points, 0) for labels in labels_list]
        bbox_targets_list = [
            bbox_targets.split(num_points, 0)
            for bbox_targets in bbox_targets_list
        ]

        # concat per level image
        concat_lvl_labels = []
        concat_lvl_bbox_targets = []
        for i in range(num_levels):
            concat_lvl_labels.append(
                torch.cat([labels[i] for labels in labels_list]))
            bbox_targets = torch.cat(
                [bbox_targets[i] for bbox_targets in bbox_targets_list])
            if self.norm_on_bbox:
                bbox_targets[:, :4] = bbox_targets[:, :4] / self.strides[i]
            concat_lvl_bbox_targets.append(bbox_targets)
        return concat_lvl_labels, concat_lvl_bbox_targets

    def _get_target_single(self, gt_bboxes, gt_labels, points, regress_ranges,
                           num_points_per_lvl):
        """Compute regression and classification targets for a single image."""
        num_points = points.size(0)
        num_gts = gt_labels.size(0)
        if num_gts == 0:
            return gt_labels.new_full((num_points,), self.background_label), \
                   gt_bboxes.new_zeros((num_points, 5))

        areas = gt_bboxes[:, 2] * gt_bboxes[:, 3]
        # TODO: figure out why these two are different
        # areas = areas[None].expand(num_points, num_gts)
        areas = areas[None].repeat(num_points, 1)
        regress_ranges = regress_ranges[:, None, :].expand(
            num_points, num_gts, 2)
        points = points[:, None, :].expand(num_points, num_gts, 2)
        gt_bboxes = mintheta_obb(gt_bboxes)
        gt_bboxes = gt_bboxes[None].expand(num_points, num_gts, 5)
        gt_ctr, gt_wh, gt_thetas = torch.split(
            gt_bboxes, [2, 2, 1], dim=2)

        Cos, Sin = torch.cos(gt_thetas), torch.sin(gt_thetas)
        Matrix = torch.cat([Cos, -Sin, Sin, Cos], dim=-1).reshape(
            num_points, num_gts, 2, 2)
        offset = points - gt_ctr
        offset = torch.matmul(Matrix, offset[..., None])
        offset = offset.squeeze(-1)

        W, H = gt_wh[..., 0], gt_wh[..., 1]
        offset_x, offset_y = offset[..., 0], offset[..., 1]
        left = W / 2 + offset_x
        right = W / 2 - offset_x
        top = H / 2 + offset_y
        bottom = H / 2 - offset_y
        bbox_targets = torch.stack((left, top, right, bottom), -1)

        # condition1: inside a gt bbox
        # if center_sampling is true, also in center bbox.
        inside_gt_bbox_mask = bbox_targets.min(-1)[0] > 0
        if self.center_sampling:
            # inside a `center bbox`
            radius = self.center_sample_radius
            stride = offset.new_zeros(offset.shape)

            # project the points on current lvl back to the `original` sizes
            lvl_begin = 0
            for lvl_idx, num_points_lvl in enumerate(num_points_per_lvl):
                lvl_end = lvl_begin + num_points_lvl
                stride[lvl_begin:lvl_end] = self.strides[lvl_idx] * radius
                lvl_begin = lvl_end

            inside_center_bbox_mask = (abs(offset) < stride).all(dim=-1)
            inside_gt_bbox_mask = torch.logical_and(
                inside_center_bbox_mask, inside_gt_bbox_mask)

        # condition2: limit the regression range for each location
        max_regress_distance = bbox_targets.max(-1)[0]
        inside_regress_range = (
            (max_regress_distance >= regress_ranges[..., 0])
            & (max_regress_distance <= regress_ranges[..., 1]))

        # if there are still more than one objects for a location,
        # we choose the one with minimal area
        areas[inside_gt_bbox_mask == 0] = INF
        areas[inside_regress_range == 0] = INF
        min_area, min_area_inds = areas.min(dim=1)

        labels = gt_labels[min_area_inds]
        labels[min_area == INF] = self.background_label  # set as BG
        bbox_targets = bbox_targets[range(num_points), min_area_inds]

        theta_targets = gt_thetas[range(num_points), min_area_inds]
        bbox_targets = torch.cat([bbox_targets, theta_targets], dim=1)
        return labels, bbox_targets

    def centerness_target(self, pos_bbox_targets):
        """Compute centerness targets for oriented bounding boxes (OBB).

        Args:
            pos_bbox_targets (Tensor): BBox targets of positive bboxes in shape
                (num_pos, 5), where the 5 columns are [d_xl, d_yt, d_xr, d_xb, d_theta].

        Returns:
            Tensor: Centerness target.
        """
        # only calculate pos centerness targets, otherwise there may be nan
        left_right = pos_bbox_targets[:, [0, 2]]
        top_bottom = pos_bbox_targets[:, [1, 3]]
        centerness_targets = (
            left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) * (
                top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
        return torch.sqrt(centerness_targets)
