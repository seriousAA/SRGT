import torch
import glob
import os
import os.path as osp
import shutil
import types
import ipdb

from mmcv.runner import BaseRunner, EpochBasedRunner, IterBasedRunner, EvalHook, _load_checkpoint
from mmcv.utils import Config

from .signature import parse_method_info
from .vars import resolve


def find_latest_checkpoint(path, ext="pth"):
    if not osp.exists(path):
        return None
    if osp.exists(osp.join(path, f"latest.{ext}")):
        return osp.join(path, f"latest.{ext}")

    checkpoints = glob.glob(osp.join(path, f"*.{ext}"))
    if len(checkpoints) == 0:
        return None
    latest = -1
    latest_path = None
    for checkpoint in checkpoints:
        count = int(osp.basename(checkpoint).split("_")[-1].split(".")[0])
        if count > latest:
            latest = count
            latest_path = checkpoint
    return latest_path


def patch_checkpoint(runner: BaseRunner):
    # patch save_checkpoint
    old_save_checkpoint = runner.save_checkpoint
    params = parse_method_info(old_save_checkpoint)
    default_tmpl = params["filename_tmpl"].default

    def save_checkpoint(self, out_dir, filename_tmpl=default_tmpl, **kwargs):
        create_symlink = kwargs.get("create_symlink", True)
        # create_symlink
        kwargs.update(create_symlink=create_symlink)
        old_save_checkpoint(out_dir, filename_tmpl=filename_tmpl, **kwargs)
        if not create_symlink:
            dst_file = osp.join(out_dir, "latest.pth")
            if isinstance(self, EpochBasedRunner):
                filename = filename_tmpl.format(self.epoch + 1)
            elif isinstance(self, IterBasedRunner):
                filename = filename_tmpl.format(self.iter + 1)
            else:
                raise NotImplementedError()
            filepath = osp.join(out_dir, filename)
            shutil.copy(filepath, dst_file)

    runner.save_checkpoint = types.MethodType(save_checkpoint, runner)
    return runner


def patch_resume(runner: BaseRunner):
    # patch resume function
    old_resume = runner.resume
    params = parse_method_info(old_resume)
    default_map = params["map_location"].default

    def resume(self, checkpoint, map_location=default_map, **kwargs):
        if map_location == 'default':
            device_id = torch.cuda.current_device()
            ckpt = _load_checkpoint(
                checkpoint,
                map_location=lambda storage, loc: storage.cuda(device_id),
                logger=self.logger)
        else:
            ckpt = _load_checkpoint(
                checkpoint, map_location=map_location)
        self.resume_start_point_iter = ckpt['meta']['iter']
        old_resume(checkpoint, map_location=map_location, **kwargs)

    runner.resume = types.MethodType(resume, runner)
    return runner


def patch_runner(runner):
    runner = patch_checkpoint(runner)
    runner = patch_resume(runner)
    return runner


def setup_env(cfg):
    os.environ["WORK_DIR"] = cfg.get("work_dir", "")


def patch_config(cfg):
    
    cfg_dict = super(Config, cfg).__getattribute__("_cfg_dict").to_dict()
    cfg_dict["cfg_name"] = osp.splitext(osp.basename(cfg.filename))[0]
    cfg_dict = resolve(cfg_dict)
    cfg = Config(cfg_dict, filename=cfg.filename)
    # wrap for semi
    if cfg.get("semi_wrapper", None):
        setattr(cfg.semi_wrapper.model, "train_cfg", cfg.pop("train_cfg", None))
        setattr(cfg.semi_wrapper.model, "test_cfg", cfg.pop("test_cfg", None))
        cfg.model = cfg.semi_wrapper
        cfg.pop("semi_wrapper")
    # enable environment variables
    setup_env(cfg)
    return cfg


def patch_eval_hook(eval_hook: EvalHook, skip_at_resume_point=False):
    # patch resume function
    old_should_evaluate = eval_hook._should_evaluate
    old_before_run = eval_hook.before_run

    def before_run(self, runner):
        self.first_skip_logged = False
        old_before_run(runner)

    def _should_evaluate(self, runner):
        if skip_at_resume_point and hasattr(runner, "resume_start_point_iter") \
            and abs(runner.iter - runner.resume_start_point_iter) <= 10:
            if not self.first_skip_logged:
                runner.logger.info(f"Skip evaluation at resume point, iter: {runner.iter}, " +
                                   f"resume_start_point_iter: {runner.resume_start_point_iter}")
                self.first_skip_logged = True
            return False
        return old_should_evaluate(runner)

    eval_hook.before_run = types.MethodType(before_run, eval_hook)
    eval_hook._should_evaluate = types.MethodType(_should_evaluate, eval_hook)
    return eval_hook

