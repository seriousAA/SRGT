import random
import warnings
import re
import numpy as np
import torch
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (
    HOOKS,
    DistSamplerSeedHook,
    EpochBasedRunner,
    Fp16OptimizerHook,
    OptimizerHook,
    build_optimizer,
    build_runner,
)
from mmcv.runner.hooks import HOOKS
from mmcv.utils import build_from_cfg
from mmdet.core import EvalHook, DistEvalHook
from mmdet.datasets import build_dataset

from ssod.datasets import build_dataloader, replace_ImageToTensor
from ssod.utils import find_latest_checkpoint, get_root_logger, patch_runner, patch_eval_hook
import ipdb

def train_detector(
    model, dataset, cfg, distributed=False, validate=False, timestamp=None, meta=None
):
    logger = get_root_logger(log_level=cfg.log_level)

    # prepare data loaders
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]
    if "imgs_per_gpu" in cfg.data:
        logger.warning(
            '"imgs_per_gpu" is deprecated in MMDet V2.0. '
            'Please use "samples_per_gpu" instead'
        )
        if "samples_per_gpu" in cfg.data:
            logger.warning(
                f'Got "imgs_per_gpu"={cfg.data.imgs_per_gpu} and '
                f'"samples_per_gpu"={cfg.data.samples_per_gpu}, "imgs_per_gpu"'
                f"={cfg.data.imgs_per_gpu} is used in this experiments"
            )
        else:
            logger.warning(
                'Automatically set "samples_per_gpu"="imgs_per_gpu"='
                f"{cfg.data.imgs_per_gpu} in this experiments"
            )
        cfg.data.samples_per_gpu = cfg.data.imgs_per_gpu
    

    runner_type = 'EpochBasedRunner' if 'runner' not in cfg else cfg.runner['type']

    train_dataloader_default_args = dict(
        samples_per_gpu=cfg.data.samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        # `num_gpus` will be ignored if distributed
        num_gpus=len(cfg.gpu_ids),
        dist=distributed,
        seed=cfg.seed,
        runner_type=runner_type,
        persistent_workers=False)

    train_loader_cfg = {
        **train_dataloader_default_args,
        **cfg.data.get('train_dataloader', {})
    }
    train_loader_cfg.update(sampler_cfg=cfg.data.get("sampler", {}).get("train", {}))

    data_loaders = [build_dataloader(ds, **train_loader_cfg) for ds in dataset]
    
    # put model on gpus
    if distributed:
        find_unused_parameters = cfg.get("find_unused_parameters", False)
        # Sets the `find_unused_parameters` parameter in
        # torch.nn.parallel.DistributedDataParallel
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters,
        )
    else:
        model = MMDataParallel(model.cuda(cfg.gpu_ids[0]), device_ids=cfg.gpu_ids)

    # build runner
    optimizer = build_optimizer(model, cfg.optimizer)

    if "runner" not in cfg:
        cfg.runner = {"type": "EpochBasedRunner", "max_epochs": cfg.total_epochs}
        warnings.warn(
            "config is now expected to have a `runner` section, "
            "please set `runner` in your config.",
            UserWarning,
        )
    else:
        if "total_epochs" in cfg and cfg.runner.type == "EpochBasedRunner":
            assert cfg.total_epochs == cfg.runner.max_epochs

    runner = build_runner(
        cfg.runner,
        default_args=dict(
            model=model,
            optimizer=optimizer,
            work_dir=cfg.work_dir,
            logger=logger,
            meta=meta,
        ),
    )

    # an ugly workaround to make .log and .log.json filenames the same
    runner.timestamp = timestamp

    # fp16 setting
    fp16_cfg = cfg.get("fp16", None)
    if fp16_cfg is not None:
        optimizer_config = Fp16OptimizerHook(
            **cfg.optimizer_config, **fp16_cfg, distributed=distributed
        )
    elif distributed and "type" not in cfg.optimizer_config:
        optimizer_config = OptimizerHook(**cfg.optimizer_config)
    else:
        optimizer_config = cfg.optimizer_config

    # register hooks
    runner.register_training_hooks(
        cfg.lr_config,
        optimizer_config,
        cfg.checkpoint_config,
        cfg.log_config,
        cfg.get("momentum_config", None),
    )
    if distributed:
        if isinstance(runner, EpochBasedRunner):
            runner.register_hook(DistSamplerSeedHook())

    # register eval hooks
    if validate:
        # Support batch_size > 1 in validation
        val_samples_per_gpu = cfg.data.val.pop("samples_per_gpu", 1)
        if val_samples_per_gpu > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.val.pipeline = replace_ImageToTensor(cfg.data.val.pipeline)
        val_dataset = build_dataset(cfg.data.val, dict(test_mode=True))
        val_dataloader = build_dataloader(
            val_dataset,
            samples_per_gpu=val_samples_per_gpu,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=distributed,
            shuffle=False,
        )
        eval_cfg = cfg.get("evaluation", {})
        eval_cfg["by_epoch"] = eval_cfg.get(
            "by_epoch", cfg.runner["type"] != "IterBasedRunner"
        )
        skip_at_resume_point = eval_cfg.pop("skip_at_resume_point", False)
        if "type" not in eval_cfg:
            eval_hook = DistEvalHook if distributed else EvalHook
            eval_hook = eval_hook(val_dataloader, **eval_cfg)

        else:
            if not distributed and "dist" in eval_cfg["type"].lower():
                UserWarning(
                    "Distributed evaluation hook is used in non-distributed "
                    "training environment"
                )
                eval_cfg["type"] = re.sub(r"dist", "", eval_cfg["type"], flags=re.IGNORECASE)
            elif distributed and "dist" not in eval_cfg["type"].lower():
                UserWarning(
                    "Non-distributed evaluation hook is used in distributed "
                    "training environment"
                )
                eval_cfg["type"] = f"Dist{eval_cfg['type']}"
            eval_hook = build_from_cfg(
                eval_cfg, HOOKS, default_args=dict(dataloader=val_dataloader)
            )
        eval_hook = patch_eval_hook(eval_hook, skip_at_resume_point)
        runner.register_hook(eval_hook, priority="LOW")

    # user-defined hooks
    if cfg.get("custom_hooks", None):
        custom_hooks = cfg.custom_hooks
        assert isinstance(
            custom_hooks, list
        ), f"custom_hooks expect list type, but got {type(custom_hooks)}"
        for hook_cfg in cfg.custom_hooks:
            assert isinstance(hook_cfg, dict), (
                "Each item in custom_hooks expects dict type, but got "
                f"{type(hook_cfg)}"
            )
            hook_cfg = hook_cfg.copy()
            priority = hook_cfg.pop("priority", "NORMAL")
            hook = build_from_cfg(hook_cfg, HOOKS)
            runner.register_hook(hook, priority=priority)

    runner = patch_runner(runner)
    resume_from = None
    if cfg.get("auto_resume", False):
        resume_from = find_latest_checkpoint(cfg.work_dir)
    if resume_from is not None:
        cfg.resume_from = resume_from

    if cfg.get("resume_from", None):
        runner.resume(cfg.resume_from, resume_optimizer=cfg.get("resume_optimizer", True))
    elif cfg.get("load_from", None):
        runner.load_checkpoint(cfg.load_from)
    elif cfg.get("resume_path", None):
        runner.resume(cfg.resume_path, resume_optimizer=False)
        runner._iter = cfg.get("resume_start", 0)
        if hasattr(runner, "resume_start_point_iter"):
            runner.resume_start_point_iter = cfg.get("resume_start", 0)
    runner.run(data_loaders, cfg.workflow)
