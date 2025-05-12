import torch
import torch.nn as nn
import math
from mmdet.models.builder import LOSSES
from mmdet.models.losses import smooth_l1_loss, weighted_loss

@LOSSES.register_module()
class NLLoss(nn.Module):
    def __init__(self, beta=1.0, nll_weight=0.05, loss_weight=1.0, reduction='mean'):
        super(NLLoss, self).__init__()
        self.beta = beta
        self.loss_weight = loss_weight
        self.nll_weight = nll_weight  # Configurable scaling factor for NLL loss
        self.reduction = reduction

    def forward(self, 
                pred_deltas, 
                pred_deltas_std, 
                target_deltas,
                iou_weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        """
        Args:
            pred_deltas (Tensor): Predicted bounding box deltas.
            pred_deltas_std (Tensor): Predicted bounding box variance (std).
            target_deltas (Tensor): Target bounding box deltas.
            iou_weight (Tensor): IoU-based weights for the loss.
            avg_factor (int): Average factor for the loss.
            reduction_override (str): Override for the reduction method.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        # Default IoU weights to 1 if not provided
        if iou_weight is None:
            iou_weight = torch.ones(target_deltas.size(0), dtype=torch.float
                                    ).to(target_deltas.device)

        # Calculate the NLL loss based on predicted deltas, variance (std), and target deltas
        nll_loss = _nl_loss(
            pred=(pred_deltas,pred_deltas_std),
            target=target_deltas,
            weight=iou_weight,
            avg_factor=avg_factor,
            reduction=reduction
        )

        # Calculate smooth L1 loss for comparison
        smooth_l1 = smooth_l1_loss(
            pred_deltas,
            target_deltas,
            weight=None,
            beta=self.beta,
            avg_factor=avg_factor,
            reduction=reduction)

        # Combine the losses (weighted)
        total_loss = smooth_l1 + self.nll_weight * nll_loss

        return self.loss_weight * total_loss

@weighted_loss
def _nl_loss(pred, target):
    """
    Compute the negative log-likelihood (NLL) loss.
    Args:
        input (Tensor): Predicted deltas.
        input_std (Tensor): Predicted standard deviations (variance).
        target (Tensor): Ground truth deltas.
        iou_weight (Tensor): IoU-based weights for the loss.
        reduction (str): Reduction method to apply to the loss.
    """
    input, input_std = pred
    sigma = input_std.sigmoid()
    sigma_sq = torch.square(sigma)
    # Ensure input sizes are consistent
    assert input.size() == input_std.size() == target.size()

    # Calculate the NLL loss using variance
    first_term = (target - input) ** 2 / (2 * sigma_sq)
    second_term = torch.log(sigma_sq) / 2
    nll_loss = first_term + second_term

    return nll_loss.sum(dim=1) + 2 * torch.log(
        2 * torch.Tensor([math.pi]).to(nll_loss.device)
    )
