import torch
import datetime
from typing import Dict
from mmcv.runner import TextLoggerHook
from mmcv.runner.hooks import HOOKS
from collections import OrderedDict
import mmcv

@HOOKS.register_module()
class SemiTextLoggerHook(TextLoggerHook):
    
    def __init__(self,
                 ignore_keys=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.ignore_keys = ignore_keys if ignore_keys is not None else []

    def _log_info(self, log_dict: Dict, runner) -> None:
        # print exp name for users to distinguish experiments
        # at every ``interval_exp_name`` iterations and the end of each epoch
        if runner.meta is not None and 'exp_name' in runner.meta:
            if (self.every_n_iters(runner, self.interval_exp_name)) or (
                    self.by_epoch and self.end_of_epoch(runner)):
                exp_info = f'Exp name: {runner.meta["exp_name"]}'
                runner.logger.info(exp_info)

        if log_dict['mode'] == 'train':
            if isinstance(log_dict['lr'], dict):
                lr_str = []
                for k, val in log_dict['lr'].items():
                    lr_str.append(f'lr_{k}: {val:.3e}')
                lr_str = ' '.join(lr_str)  # type: ignore
            else:
                lr_str = f'lr: {log_dict["lr"]:.3e}'  # type: ignore

            # by epoch: Epoch [4][100/1000]
            # by iter:  Iter [100/100000]
            if self.by_epoch:
                log_str = f'Epoch [{log_dict["epoch"]}]' \
                          f'[{log_dict["iter"]}/{len(runner.data_loader)}]\t'
            else:
                log_str = f'Iter [{log_dict["iter"]}/{runner.max_iters}]\t'
            log_str += f'{lr_str}, '

            if 'time' in log_dict.keys():
                self.time_sec_tot += (log_dict['time'] * self.interval)
                time_sec_avg = self.time_sec_tot / (
                    runner.iter - self.start_iter + 1)
                eta_sec = time_sec_avg * (runner.max_iters - runner.iter - 1)
                eta_str = str(datetime.timedelta(seconds=int(eta_sec)))
                log_str += f'eta: {eta_str}, '
                log_str += f'time: {log_dict["time"]:.3f}, ' \
                           f'data_time: {log_dict["data_time"]:.3f}, '
                # statistic memory
                if torch.cuda.is_available():
                    log_str += f'memory: {log_dict["memory"]}, '
        else:
            # val/test time
            # here 1000 is the length of the val dataloader
            # by epoch: Epoch[val] [4][1000]
            # by iter: Iter[val] [1000]
            if self.by_epoch:
                log_str = f'Epoch({log_dict["mode"]}) ' \
                    f'[{log_dict["epoch"]}][{log_dict["iter"]}]\t'
            else:
                log_str = f'Iter({log_dict["mode"]}) [{log_dict["iter"]}]\t'

        log_items = []
        loss_val = None
        losses_val = None
        grad_norm_val = None

        for name, val in log_dict.items():
            if any(name.startswith(key) for key in self.ignore_keys):
                continue
            # TODO: resolve this hack
            # these items have been in log_str
            if name in [
                    'mode', 'Epoch', 'iter', 'lr', 'time', 'data_time',
                    'memory', 'epoch'
            ]:
                continue
            if isinstance(val, float):
                # 4 decimal places minimum, 6 maximum
                val = f'{val:.4f}' if abs(val) >= 1e-4 else f'{val:.6f}'
            
            # Save loss, losses, and grad_norm for later placement
            if name == 'loss':
                loss_val = f'{name}: {val}'
            elif name == 'losses':
                losses_val = f'{name}: {val}'
            elif name == 'grad_norm':
                grad_norm_val = f'{name}: {val}'
            else:
                log_items.append(f'{name}: {val}')

        # If loss or losses exist, make sure they are placed as second-to-last item
        if loss_val is not None:
            log_items.append(loss_val)
        elif losses_val is not None:
            log_items.append(losses_val)

        # Finally append grad_norm if it exists, as the last item
        if grad_norm_val is not None:
            log_items.append(grad_norm_val)

        log_str += ', '.join(log_items)

        runner.logger.info(log_str)

    def _dump_log(self, log_dict: Dict, runner) -> None:
        # dump log in json format
        json_log = OrderedDict()
        for k, v in log_dict.items():
            if any(k.startswith(key) for key in self.ignore_keys):
                continue
            json_log[k] = self._round_float(v)
        # only append log at last line
        if runner.rank == 0:
            with open(self.json_log_path, 'a+') as f:
                mmcv.dump(json_log, f, file_format='json')
                f.write('\n')
