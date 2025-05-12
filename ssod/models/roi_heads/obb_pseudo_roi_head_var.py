import torch

from mmdet.core import arb2result, arb2roi, build_assigner, build_sampler
from mmdet.models.builder import HEADS, build_head
from mmdet.models.roi_heads import OBBStandardRoIHead

@HEADS.register_module()
class OBBPseudoRoIHeadWithVar(OBBStandardRoIHead):
    def _bbox_forward(self, x, rois):
        """Box head forward function used in both training and testing"""
        # Extract bounding box features
        bbox_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], rois)
        if self.with_shared_head:
            bbox_feats = self.shared_head(bbox_feats)
        
        # Get classification scores, bbox predictions, and bbox std predictions
        cls_score, bbox_pred, bbox_pred_std = self.bbox_head(bbox_feats)

        bbox_results = dict(
            cls_score=cls_score,
            bbox_pred=bbox_pred,
            bbox_pred_std=bbox_pred_std,  # Include the bbox_pred_std output
            bbox_feats=bbox_feats)
        return bbox_results

    def _bbox_forward_train(self, x, sampling_results, gt_bboxes, gt_labels,
                            img_metas=None, **kwargs):
        """Run forward function and calculate loss for box head in training"""
        rois = kwargs.pop('rois', None)
        if rois is None:
            rois = arb2roi([res.bboxes for res in sampling_results],
                        bbox_type=self.bbox_head.start_bbox_type)
        
        # Get the bbox predictions along with the variance predictions
        bbox_results = self._bbox_forward(x, rois)

        # Get the bbox targets
        bbox_targets = self.bbox_head.get_targets(
            sampling_results, gt_bboxes, gt_labels, self.train_cfg)

        # Calculate the loss with bbox_pred_std integrated
        losses = self.bbox_head.loss(
            bbox_results['cls_score'],
            bbox_results['bbox_pred'],
            bbox_results['bbox_pred_std'],  # Pass bbox_pred_std to the loss function
            rois,
            *bbox_targets,
            **kwargs
        )

        bbox_results.update(loss_bbox=losses)
        return bbox_results

    def simple_test_bboxes(self,
                           x,
                           img_metas,
                           proposals,
                           rcnn_test_cfg,
                           rescale=False,
                           return_var=False):
        """Test only det bboxes without augmentation.

        Args:
            x (tuple[Tensor]): Feature maps of all scale level.
            img_metas (list[dict]): Image meta info.
            proposals (List[Tensor]): Region proposals.
            rcnn_test_cfg (obj:`ConfigDict`): `test_cfg` of R-CNN.
            rescale (bool): If True, return boxes in original image space.
                Default: False.

        Returns:
            tuple[list[Tensor], list[Tensor]]: The first list contains
                the boxes of the corresponding image in a batch, each
                tensor has the shape (num_boxes, 5) and last dimension
                5 represent (tl_x, tl_y, br_x, br_y, score). Each Tensor
                in the second list is the labels with shape (num_boxes, ).
                The length of both lists should be equal to batch_size.
        """
        rois = arb2roi(proposals, bbox_type=self.bbox_head.start_bbox_type)
        if rois.shape[0] == 0:
            batch_size = len(proposals)
            det_bbox = rois.new_zeros(0, 6)
            det_label = rois.new_zeros((0, ), dtype=torch.long)
            if rcnn_test_cfg is None:
                det_bbox = det_bbox[:, :5]
                det_label = rois.new_zeros(
                    (0, self.bbox_head.fc_cls.out_features))
            # There is no proposal in the whole batch
            return [det_bbox] * batch_size, [det_label] * batch_size

        bbox_results = self._bbox_forward(x, rois)
        img_shapes = tuple(meta['img_shape'] for meta in img_metas)
        scale_factors = tuple(meta['scale_factor'] for meta in img_metas)

        # split batch bbox prediction back to each image
        cls_score = bbox_results['cls_score']
        bbox_pred = bbox_results['bbox_pred']
        bbox_pred_std = bbox_results['bbox_pred_std']
        num_proposals_per_img = tuple(len(p) for p in proposals)
        rois = rois.split(num_proposals_per_img, 0)
        cls_score = cls_score.split(num_proposals_per_img, 0)

        # some detector with_reg is False, bbox_pred will be None
        if bbox_pred is not None:
            # TODO move this to a sabl_roi_head
            # the bbox prediction of some detectors like SABL is not Tensor
            if isinstance(bbox_pred, torch.Tensor):
                bbox_pred = bbox_pred.split(num_proposals_per_img, 0)
                bbox_pred_std = bbox_pred_std.split(num_proposals_per_img, 0)
            else:
                bbox_pred = self.bbox_head.bbox_pred_split(
                    bbox_pred, num_proposals_per_img)
                bbox_pred_std = self.bbox_head.bbox_pred_split(
                    bbox_pred_std, num_proposals_per_img)
        else:
            bbox_pred = (None, ) * len(proposals)
            bbox_pred_std = (None, ) * len(proposals)

        # apply bbox post-processing to each image individually
        det_bboxes = []
        det_labels = []
        det_bboxes_std = []
        for i in range(len(proposals)):
            if rois[i].shape[0] == 0:
                # There is no proposal in the single image
                det_bbox = rois[i].new_zeros(0, 6)
                det_label = rois[i].new_zeros((0, ), dtype=torch.long)
                if rcnn_test_cfg is None:
                    det_bbox = det_bbox[:, :5]
                    det_label = rois[i].new_zeros(
                        (0, self.bbox_head.fc_cls.out_features))

            else:
                det_bbox, det_label, det_bbox_std = self.bbox_head.get_bboxes(
                    rois[i],
                    cls_score[i],
                    bbox_pred[i],
                    img_shapes[i],
                    scale_factors[i],
                    bbox_pred_std=bbox_pred_std[i],
                    rescale=rescale,
                    cfg=rcnn_test_cfg)
            det_bboxes.append(det_bbox)
            det_labels.append(det_label)
            det_bboxes_std.append(det_bbox_std)
        if return_var:
            return det_bboxes, det_labels, det_bboxes_std
        return det_bboxes, det_labels

    def aug_test_bboxes(self, feats, img_metas, proposal_list, rcnn_test_cfg):
        """Test det bboxes with test time augmentation."""
        pass