#!/usr/bin/env python3
import os
import sys
import time
import logging
import datetime
from typing import Dict, Any, Optional, Union, List
from pathlib import Path
import json
from contextlib import contextmanager

import numpy as np
import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


class ColoredFormatter(logging.Formatter):
    COLORS = {
        'DEBUG': '\033[36m',    
        'INFO': '\033[32m',     
        'WARNING': '\033[33m',  
        'ERROR': '\033[31m',   
        'CRITICAL': '\033[35m',
        'RESET': '\033[0m' 
    }
    
    def format(self, record):
        color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        record.levelname = f"{color}{record.levelname}{self.COLORS['RESET']}"
        return super().format(record)


class TrainingLogger:

    def __init__(self, 
                 name: str,
                 log_dir: str,
                 exp_index: int,
                 enable_tensorboard: bool = True,
                 console_level: str = "INFO",
                 file_level: str = "DEBUG"):
        self.name = name
        self.log_dir = Path(log_dir)
        self.exp_index = exp_index
        self.start_time = time.time()

        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.logger = logging.getLogger(f"{name}_{exp_index}")
        self.logger.setLevel(logging.DEBUG)
        self.logger.handlers.clear()

        log_file = self.log_dir / f"{name}_{exp_index}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(getattr(logging, file_level.upper()))
        file_formatter = logging.Formatter(
            '%(asctime)s | %(name)s | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, console_level.upper()))
        console_formatter = ColoredFormatter(
            '%(asctime)s | %(levelname)s | %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)

        self.tb_writer = None
        if enable_tensorboard:
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            tb_dir = self.log_dir.parent / "TensorBoardLogs" / f"exp{exp_index}_{timestamp}"
            tb_dir.mkdir(parents=True, exist_ok=True)
            self.tb_writer = SummaryWriter(str(tb_dir))

        self.train_stats = {
            'total_steps': 0,
            'best_metric': float('inf'),
            'best_step': 0,
            'epoch_start_time': None,
            'step_times': [],
            'losses': [],
            'metrics_history': []
        }
        
        self.info(f"=== Experiment ID: {exp_index} | Log file: {log_file.name} ===")
    
    def info(self, message: str, **kwargs):

        self.logger.info(message)
        if kwargs and self.tb_writer:
            step = kwargs.get('step', self.train_stats['total_steps'])
            for key, value in kwargs.items():
                if key != 'step' and isinstance(value, (int, float)):
                    self.tb_writer.add_scalar(f"info/{key}", value, step)
    
    def debug(self, message: str):

        self.logger.debug(message)
    
    def warning(self, message: str):

        self.logger.warning(message)
    
    def error(self, message: str):

        self.logger.error(message)
    
    def log_training_config(self, config: Dict[str, Any]):

        self.info("=== Training Configuration ===")
        for key, value in config.items():
            self.info(f"  {key}: {value}")
    
    def log_model_info(self, model, model_name: str):

        param_count = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        if self.tb_writer:
            self.tb_writer.add_scalar("model/total_params", param_count, 0)
            self.tb_writer.add_scalar("model/trainable_params", trainable_params, 0)
    
    def log_step(self, step: int, loss: float, 
                 metrics: Optional[Dict[str, float]] = None,
                 lr: Optional[float] = None,
                 show_progress: bool = True):
        self.train_stats['total_steps'] = step
        self.train_stats['losses'].append(loss)

        current_time = time.time()
        if self.train_stats['step_times']:
            step_time = current_time - self.train_stats['step_times'][-1]
        else:
            step_time = 0.0
        self.train_stats['step_times'].append(current_time)

        log_parts = [f"Step {step:6d}"]
        log_parts.append(f"Loss: {loss:.6f}")
        
        if lr is not None:
            log_parts.append(f"LR: {lr:.2e}")
        
        if step_time > 0:
            log_parts.append(f"Time: {step_time:.2f}s")
            steps_per_sec = 1.0 / step_time if step_time > 0 else 0
            log_parts.append(f"Speed: {steps_per_sec:.2f} step/s")
        
        if metrics:
            for key, value in metrics.items():
                log_parts.append(f"{key}: {value:.6f}")
        
        message = " | ".join(log_parts)
        
        if show_progress:
            self.info(message)
        else:
            self.debug(message)
        
        if self.tb_writer:
            self.tb_writer.add_scalar("train/loss", loss, step)
            if lr is not None:
                self.tb_writer.add_scalar("train/learning_rate", lr, step)
            if step_time > 0:
                self.tb_writer.add_scalar("train/step_time", step_time, step)
            if metrics:
                for key, value in metrics.items():
                    self.tb_writer.add_scalar(f"train/{key}", value, step)
    
    def log_evaluation(self, step: int, eval_metrics: Dict[str, float], 
                      is_best: bool = False):

        self.train_stats['metrics_history'].append({
            'step': step,
            'metrics': eval_metrics.copy(),
            'is_best': is_best
        })

        if 'mae' in eval_metrics:
            if eval_metrics['mae'] < self.train_stats['best_metric']:
                self.train_stats['best_metric'] = eval_metrics['mae']
                self.train_stats['best_step'] = step
                is_best = True

        metric_parts = []
        for key, value in eval_metrics.items():
            if isinstance(value, (list, np.ndarray)):
                if len(value) <= 6:
                    metric_parts.append(f"{key}: {[f'{v:.3f}' for v in value]}")
                else:
                    metric_parts.append(f"{key}: [{value[0]:.3f}...{value[-1]:.3f}] (len={len(value)})")
            else:
                metric_parts.append(f"{key}: {value:.6f}")
        
        status = "NEW BEST" if is_best else "EVAL"
        message = f"{status} | Step {step} | " + " | ".join(metric_parts)
        self.info(message)

        if self.tb_writer:
            for key, value in eval_metrics.items():
                if isinstance(value, (int, float)):
                    self.tb_writer.add_scalar(f"eval/{key}", value, step)
                elif isinstance(value, (list, np.ndarray)):
                    for i, v in enumerate(value):
                        self.tb_writer.add_scalar(f"eval/{key}_step_{i+1}", v, step)
    
    def log_epoch_summary(self, epoch: int, avg_loss: float, 
                         epoch_time: float, total_epochs: int):

        remaining_epochs = total_epochs - epoch
        remaining_time = remaining_epochs * epoch_time if epoch_time > 0 else 0
        
        self.info("=" * 80)
        self.info(f"Epoch {epoch:4d}/{total_epochs} Completed")
        self.info(f"Avg Loss: {avg_loss:.6f}")
        self.info(f"Cost Time: {epoch_time:.2f}s")
        self.info(f"Remaining Time: {self._format_time(remaining_time)}")
        self.info(f"Best Metric: {self.train_stats['best_metric']:.6f} (Step {self.train_stats['best_step']})")
        self.info("=" * 80)
        
        if self.tb_writer:
            self.tb_writer.add_scalar("epoch/avg_loss", avg_loss, epoch)
            self.tb_writer.add_scalar("epoch/time", epoch_time, epoch)
    
    def create_progress_bar(self, total: int, desc: str = "Training", **kwargs) -> tqdm:
        return tqdm(
            total=total,
            desc=f"🚀 {desc}",
            ncols=120,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}',
            **kwargs
        )
    
    def _format_time(self, seconds: float) -> str:
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            return f"{seconds/60:.1f}min"
        else:
            return f"{seconds/3600:.1f}h"
    
    def close(self):
        total_time = time.time() - self.start_time

        if self.tb_writer:
            self.tb_writer.close()


@contextmanager
def training_session(name: str, log_dir: str, exp_index: int, **kwargs):

    logger = TrainingLogger(name, log_dir, exp_index, **kwargs)
    try:
        yield logger
    finally:
        logger.close()


class DiffusionTrainingLogger(TrainingLogger):

    def log_sampling_progress(self, step: int, diffusion_step: int, total_steps: int):

        progress = (diffusion_step + 1) / total_steps * 100
        self.debug(f"Sampling Step {step} | Diffusion {diffusion_step+1}/{total_steps} ({progress:.1f}%)")
    
    def log_params_mae(self, step: int, mae: float, params_shape: tuple):

        self.info(f" Parameter Generation | Step {step} | MAE: {mae:.6f} | Shape: {params_shape}")
        if self.tb_writer:
            self.tb_writer.add_scalar("generation/params_mae", mae, step)


class PretrainLogger(TrainingLogger):
    
    def log_node_training(self, node_id: int, total_nodes: int, 
                         metrics: Dict[str, float]):

        progress = (node_id + 1) / total_nodes * 100
        metric_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        self.info(f"Node {node_id:3d}/{total_nodes} ({progress:5.1f}%) | {metric_str}")


def setup_distributed_logging(rank: int, world_size: int, logger: TrainingLogger):

    if rank == 0:
        logger.info(f"Distributed training started | Number of GPUs: {world_size}")
    else:
        logger.logger.setLevel(logging.ERROR)


if __name__ == "__main__":
    with training_session("test", "./logs", 999) as logger:
        logger.log_training_config({"lr": 1e-4, "batch_size": 32})
        logger.log_step(1, 0.5, {"acc": 0.8})
        logger.log_evaluation(1, {"mae": 0.3, "rmse": 0.4})