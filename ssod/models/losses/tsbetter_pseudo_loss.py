from mmdet.models.builder import LOSSES
from mmdet.models.losses import l1_loss, smooth_l1_loss
import torch.nn as nn
import torch

@LOSSES.register_module()
class TSBetterPseudoLoss(nn.Module):
    def __init__(self, ts_better=0.1, t_cert=0.5, loss_weight=1.0, reduction='mean'):
        super(TSBetterPseudoLoss, self).__init__()
        self.ts_better = ts_better
        self.t_cert = t_cert
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, 
                pred_deltas, 
                pred_deltas_std, 
                target_deltas,
                gt_loc_std,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        """
        Args:
            pred_deltas (Tensor): Predicted bounding box deltas.
            pred_deltas_std (Tensor): Predicted bounding box variance (std).
            target_deltas (Tensor): Target bounding box deltas.
            gt_loc_std (Tensor): Ground truth location uncertainty (std).
            avg_factor (int): Average factor for the loss.
            reduction_override (str): Override for the reduction method.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = reduction_override if reduction_override else self.reduction

        # Calculate location confidence based on the predicted std and GT std
        gt_bbox_loc_conf = 1 - gt_loc_std.sigmoid()
        pred_bbox_loc_conf = 1 - pred_deltas_std.sigmoid()

        # Determine indices where the Teacher model is better (TSBETTER logic)
        tchbetter_idx = (gt_bbox_loc_conf > pred_bbox_loc_conf + self.ts_better) & (
            gt_bbox_loc_conf > self.t_cert
        )

        # Filter out the predictions and targets based on the TSBETTER condition
        filtered_pred_deltas = pred_deltas[tchbetter_idx]
        filtered_target_deltas = target_deltas[tchbetter_idx]

        if filtered_pred_deltas.numel() == 0:  # Avoid empty selections
            return torch.tensor(0.0, device=pred_deltas.device)

        # Calculate the Smooth L1 loss for the filtered predictions
        loss = l1_loss(
            filtered_pred_deltas,
            filtered_target_deltas,
            weight=None,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        # Apply the loss weight
        return self.loss_weight * loss
