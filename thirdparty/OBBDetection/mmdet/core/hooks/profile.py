from mmcv.runner.hooks import HOOKS, Hook
try:
    from torch.utils.tensorboard import SummaryWriter       # type: ignore
except ImportError:
    SummaryWriter = None
    print("SummaryWriter could not be imported. Ensure that torch and tensorboard are installed.")
import torch
import numpy as np

@HOOKS.register_module()
class ProfileRecorder(Hook):
    def __init__(self, log_dir='./logs', log_freq=10):
        """
        Initializes the TensorBoard writer and sets up the logging directory.

        Args:
            log_dir (str): Directory where TensorBoard logs should be saved.
            log_freq (int): Frequency of logging the data per iteration.
        """
        self.writer = SummaryWriter(log_dir)
        self.log_freq = log_freq
        # self.log_graph = log_graph
        # self.graph_logged = False

    def before_run(self, runner):
        """
        Log initial setup information.
        """
        print("Starting the run...")
        self.writer.add_text('info', 'Training started', 0)

    def after_train_iter(self, runner):
        """
        Optionally logs the model graph at the first training iteration and other metrics.
        """
        # Regular logging tasks
        self.log_metrics(runner, 'train')
        if torch.cuda.is_available():
            self.log_gpu_memory(runner, 'train')

    def after_val_iter(self, runner):
        """
        Log metrics and optionally GPU memory after each validation iteration.
        """
        if runner.iter % self.log_freq == 0:
            self.log_metrics(runner, 'val')
            if torch.cuda.is_available():
                self.log_gpu_memory(runner, 'val')

    def log_metrics(self, runner, phase):
        """
        Logs various training metrics to TensorBoard.

        Args:
            runner: The current runner of the training loop.
            phase (str): Current phase of training ('train' or 'val').
        """
        loss = runner.outputs['loss']
        self.writer.add_scalar(f'{phase}/loss', loss, runner.iter)

    def log_gpu_memory(self, runner, phase):
        """
        Logs GPU memory allocation to TensorBoard.

        Args:
            runner: The current runner of the training loop.
            phase (str): Current phase of training ('train' or 'val').
        """
        allocated_memory = torch.cuda.memory_allocated()
        max_allocated_memory = torch.cuda.max_memory_allocated()
        cached_memory = torch.cuda.memory_reserved()
        self.writer.add_scalar(f'{phase}/gpu_memory/allocated', allocated_memory / (1024 ** 3), runner.iter)
        self.writer.add_scalar(f'{phase}/gpu_memory/max_allocated', max_allocated_memory / (1024 ** 3), runner.iter)
        self.writer.add_scalar(f'{phase}/gpu_memory/cached', cached_memory / (1024 ** 3), runner.iter)
        self.writer.add_scalar(f'{phase}/gpu_memory/cached-allocated', (cached_memory - allocated_memory) / (1024 ** 3), runner.iter)
        self.writer.add_scalar(f'{phase}/gpu_memory/cached-max_allocated', (cached_memory - max_allocated_memory) / (1024 ** 3), runner.iter)
        self.writer.add_scalar(f'{phase}/gpu_memory/allocated_max_allocated_avg', (allocated_memory + max_allocated_memory) / (2 * 1024 ** 3), runner.iter)
        
        # Log detailed memory summary
        memory_summary = torch.cuda.memory_summary()
        self.writer.add_text(f'{phase}/gpu_memory_summary', memory_summary, runner.iter)

        # Record detailed GPU memory stats
        memory_stats = torch.cuda.memory_stats()
        for key, value in memory_stats.items():
            self.writer.add_scalar(f'{phase}/memory_stats/{key}', value, runner.iter)

        # Log model parameter sizes and memory usage
        # for name, param in runner.model.named_parameters():
        #     param_size = param.size()
        #     param_memory = param.element_size() * param.nelement() / (1024 ** 2)  # Memory in MB
        #     self.writer.add_text(f'{phase}/params/{name}/size', str(param_size), runner.iter)
        #     self.writer.add_scalar(f'{phase}/params/{name}/memory', param_memory, runner.iter)

        # import gc
        # # Run garbage collection
        # gc.collect()
        # torch.cuda.empty_cache()

    def after_run(self, runner):
        """
        Cleanup after training is complete.
        """
        print("Run completed.")
        self.writer.add_text('info', 'Training completed', runner.iter)
        self.writer.close()
