from typing import Dict
from collections import OrderedDict

import numpy as np
import torch
import torch.distributed as dist
import os
import copy

from mmdet.models import BaseDetector, TwoStageDetector
from mmcv.runner.fp16_utils import force_fp32
from ssod.models.utils import Transform2D
from mmdet.core import bbox2type

class MultiStreamDetector(BaseDetector):
    def __init__(
        self, model: Dict[str, TwoStageDetector], train_cfg=None, test_cfg=None
    ):
        super(MultiStreamDetector, self).__init__()
        self.submodules = list(model.keys())
        for k, v in model.items():
            setattr(self, k, v)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.inference_on = self.test_cfg.get("inference_on", self.submodules[0])

    def model(self, **kwargs) -> TwoStageDetector:
        if "submodule" in kwargs:
            assert (
                kwargs["submodule"] in self.submodules
            ), "Detector does not contain submodule {}".format(kwargs["submodule"])
            model: TwoStageDetector = getattr(self, kwargs["submodule"])
        else:
            model: TwoStageDetector = getattr(self, self.inference_on)
        return model

    def freeze(self, model_ref: str):
        assert model_ref in self.submodules
        model = getattr(self, model_ref)
        model.eval()
        for param in model.parameters():
            param.requires_grad = False

    def forward_train(self, img, img_metas, **kwargs):
        super().forward_train(img, img_metas, **kwargs)
        self.freeze("teacher") # It is a must, because the freezing op in the init func may
                                # get cancelled by the MMDetection before each training iteration.

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

    def convert_bbox_space(self, img_metas_A, img_metas_B, bbox_type, bboxes_A, no_shape_filter=False):
        """ 
            function: convert bboxes_A from space A into space B
            Parameters: 
                img_metas: list(dict); bboxes_A: list(tensors)
        """
        if len(bboxes_A)==0:
            return bboxes_A
        transMat_A = [torch.from_numpy(meta["transform_matrix"]).float().to(bboxes_A[0].device)
                                for meta in img_metas_A]
        transMat_B = [torch.from_numpy(meta["transform_matrix"]).float().to(bboxes_A[0].device)
                            for meta in img_metas_B]
        M = self._get_trans_mat(transMat_A, transMat_B)
        bboxes_B, valid_masks = self._transform_bbox(
            bbox_type, bboxes_A,
            M,
            None if no_shape_filter else [meta["img_shape"] for meta in img_metas_B],
        )
        return bboxes_B, valid_masks

    @staticmethod
    def bbox2type_with_score(bboxes, to_type):
        return torch.cat([bbox2type(bboxes[..., :-1], to_type), bboxes[..., -1, None]], dim=-1)

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
