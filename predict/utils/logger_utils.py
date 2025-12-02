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
import zipfile
import xlrd
import xlwt
from sklearn.metrics import mean_absolute_error, mean_squared_error

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
            tb_dir = self.log_dir.parent / "TensorBoardLogs" / f"exp{exp_index}"
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

        self.info(f"=== Training logger initialized | Experiment ID: {exp_index} | Log file: {log_file.name} ===")

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
        config_str = json.dumps(config, indent=2, ensure_ascii=False)
        self.info(f"Configuration:\n{config_str}")
        config_file = self.log_dir / f"config_exp{self.exp_index}.json"
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

    def log_model_info(self, model, model_name: str):
        param_count = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        self.info(f"=== Model Information: {model_name} ===")
        self.info(f"Parameter count: {param_count:,}")
        self.info(f"Trainable params: {trainable_params:,}")

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
                        self.tb_writer.add_scalar(f"eval/{key}_step_{i + 1}", v, step)

    def log_epoch_summary(self, epoch: int, avg_loss: float,
                          epoch_time: float, total_epochs: int):
        remaining_epochs = total_epochs - epoch
        remaining_time = remaining_epochs * epoch_time if epoch_time > 0 else 0

        self.info("=" * 80)
        self.info(f"Epoch {epoch:4d}/{total_epochs} complete")
        self.info(f"Avg Loss: {avg_loss:.6f}")
        self.info(f"Cost Time: {epoch_time:.2f}s")
        self.info(f"Remaining time: {self._format_time(remaining_time)}")
        self.info(f"Best Metric: {self.train_stats['best_metric']:.6f} (Step {self.train_stats['best_step']})")
        self.info("=" * 80)

        if self.tb_writer:
            self.tb_writer.add_scalar("epoch/avg_loss", avg_loss, epoch)
            self.tb_writer.add_scalar("epoch/time", epoch_time, epoch)

    def create_progress_bar(self, total: int, desc: str = "Training", **kwargs) -> tqdm:
        return tqdm(
            total=total,
            desc=f"{desc}",
            ncols=120,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}',
            **kwargs
        )

    def _format_time(self, seconds: float) -> str:

        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            return f"{seconds / 60:.1f}min"
        else:
            return f"{seconds / 3600:.1f}h"

    def close(self):

        total_time = time.time() - self.start_time
        self.info(f"=== Training Complete | Cost Time: {self._format_time(total_time)} ===")

        if self.tb_writer:
            self.tb_writer.close()

        stats_file = self.log_dir / f"train_stats_exp{self.exp_index}.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            stats_to_save = self.train_stats.copy()
            if stats_to_save['metrics_history']:
                for entry in stats_to_save['metrics_history']:
                    for key, value in entry['metrics'].items():
                        if isinstance(value, np.ndarray):
                            entry['metrics'][key] = value.tolist()
            json.dump(stats_to_save, f, indent=2, ensure_ascii=False)


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
        self.debug(f"Sampling Step {step} | Diffusion {diffusion_step + 1}/{total_steps} ({progress:.1f}%)")

    def log_params_mae(self, step: int, mae: float, params_shape: tuple):
        self.info(f"Parameter Generation | Step {step} | MAE: {mae:.6f} | Shape: {params_shape}")
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
        logger.info(f"Start parallel training: {world_size}")
    else:
        logger.logger.setLevel(logging.ERROR)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def metric_func(pred, y, times):
    result = {}

    result['MSE'], result['RMSE'], result['MAE'], result['MAPE'] = np.zeros(times), np.zeros(times), np.zeros(
        times), np.zeros(times)

    def cal_MAPE(pred, y):
        diff = np.abs(np.array(y) - np.array(pred))
        return np.mean(diff / y)

    for i in range(times):
        y_i = y[:, i, :]
        pred_i = pred[:, i, :]
        MSE = mean_squared_error(pred_i, y_i)
        RMSE = mean_squared_error(pred_i, y_i) ** 0.5
        MAE = mean_absolute_error(pred_i, y_i)
        # MAPE = cal_MAPE(pred_i, y_i)
        result['MSE'][i] += MSE
        result['RMSE'][i] += RMSE
        result['MAE'][i] += MAE
        # result['MAPE'][i] += MAPE
    return result


def outputExcel(total_RMSE, total_MAE, total_MAPE):
    wb = xlwt.Workbook()
    ws = wb.add_sheet('OUTPUT')
    for i in range(6):
        ws.write(0, i, total_MAE[i])
        ws.write(1, i, total_MAPE[i] * 100)
        ws.write(2, i, total_RMSE[i])
    wb.save('test.xls')


def result_print(result, logger, info_name='Evaluate'):
    total_MSE, total_RMSE, total_MAE, total_MAPE = result['MSE'], result['RMSE'], result['MAE'], result['MAPE']

    outputExcel(total_RMSE, total_MAE, total_MAPE)

    logger.info("========== {} results ==========".format(info_name))
    logger.info(" MAE: %.3f/ %.3f/ %.3f/ %.3f/ %.3f/ %.3f" % (total_MAE[0], total_MAE[1], total_MAE[2], total_MAE[3],
                                                              total_MAE[4], total_MAE[5]))
    # logger.info("MAPE: %.3f/ %.3f/ %.3f/ %.3f/ %.3f/ %.3f"%(total_MAPE[0] * 100, total_MAPE[1] * 100, total_MAPE[2] * 100, total_MAPE[3] * 100, total_MAPE[4] * 100, total_MAPE[5] * 100))
    logger.info(
        "RMSE: %.3f/ %.3f/ %.3f/ %.3f/ %.3f/ %.3f" % (total_RMSE[0], total_RMSE[1], total_RMSE[2], total_RMSE[3],
                                                      total_RMSE[4], total_RMSE[5]))
    logger.info("---------------------------------------")

    if info_name == 'Best':
        logger.info("========== Best results ==========")
        logger.info(
            " MAE: %.3f/ %.3f/ %.3f/ %.3f/ %.3f/ %.3f" % (total_MAE[0], total_MAE[1], total_MAE[2], total_MAE[3],
                                                          total_MAE[4], total_MAE[5]))
        # logger.info("MAPE: %.3f/ %.3f/ %.3f/ %.3f/ %.3f/ %.3f"%(total_MAPE[0] * 100, total_MAPE[1] * 100, total_MAPE[2] * 100, total_MAPE[3] * 100, total_MAPE[4] * 100, total_MAPE[5] * 100))
        logger.info(
            "RMSE: %.3f/ %.3f/ %.3f/ %.3f/ %.3f/ %.3f" % (total_RMSE[0], total_RMSE[1], total_RMSE[2], total_RMSE[3],
                                                          total_RMSE[4], total_RMSE[5]))
        logger.info("---------------------------------------")


def load_data(dataset_name, stage):
    print("INFO: load {} data @ {} stage".format(dataset_name, stage))

    A = np.load("data/" + dataset_name + "/matrix.npy")
    A = get_normalized_adj(A)
    A = torch.from_numpy(A)
    X = np.load("data/" + dataset_name + "/dataset.npy")
    X = X.transpose((1, 2, 0))
    X = X.astype(np.float32)

    # Normalization using Z-score method
    means = np.mean(X, axis=(0, 2))
    X = X - means.reshape(1, -1, 1)
    stds = np.std(X, axis=(0, 2))
    X = X / stds.reshape(1, -1, 1)

    # train: 70%, validation: 10%, test: 20%
    # source: 100%, target_1day: 288, target_3day: 288*3, target_1week: 288*7
    if stage == 'train':
        X = X[:, :, :int(X.shape[2] * 0.7)]
    elif stage == 'validation':
        X = X[:, :, int(X.shape[2] * 0.7):int(X.shape[2] * 0.8)]
    elif stage == 'test':
        X = X[:, :, int(X.shape[2] * 0.8):]
    elif stage == 'source':
        X = X
    elif stage == 'target_1day':
        X = X[:, :, :288]
    elif stage == 'target_3day':
        X = X[:, :, :288 * 3]
    elif stage == 'target_1week':
        X = X[:, :, :288 * 7]
    else:
        print("Error: unsupported data stage")

    print("INFO: A shape is {}, X shape is {}, means = {}, stds = {}".format(A.shape, X.shape, means, stds))

    return A, X, means, stds


def get_normalized_adj(A):
    """
    Returns the degree normalized adjacency matrix.
    """
    A = A + np.diag(np.ones(A.shape[0], dtype=np.float32))
    D = np.array(np.sum(A, axis=1)).reshape((-1,))
    D[D <= 10e-5] = 10e-5  # Prevent infs
    diag = np.reciprocal(np.sqrt(D))
    A_wave = np.multiply(np.multiply(diag.reshape((-1, 1)), A),
                         diag.reshape((1, -1)))
    return A_wave


def generate_dataset(X, num_timesteps_input, num_timesteps_output, means, stds):
    indices = [(i, i + (num_timesteps_input + num_timesteps_output)) for i
               in range(X.shape[2] - (
                num_timesteps_input + num_timesteps_output) + 1)]
    features, target = [], []
    for i, j in indices:
        features.append(
            X[:, :, i: i + num_timesteps_input].transpose(
                (0, 2, 1)))
        target.append(X[:, 0, i + num_timesteps_input: j])

    return torch.from_numpy(np.array(features)), \
        torch.from_numpy(np.array(target))


def setup_logger(mode, DATA, index, model, aftername, node_range=None):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    c_handler = logging.StreamHandler(sys.stdout)
    log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Logs")
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

    if node_range:
        log_filename = f"{mode}_logs_{DATA}_{model}_{aftername}_{node_range}_{timestamp}.log"
    else:
        log_filename = f"{mode}_logs_{DATA}_{model}_{aftername}_{timestamp}.log"

    log_path = os.path.join(log_dir, log_filename)
    f_handler = logging.FileHandler(log_path)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s',
                                datefmt='%Y-%m-%d %H:%M:%S')
    c_handler.setFormatter(formatter)
    f_handler.setFormatter(formatter)

    logger.addHandler(c_handler)
    logger.addHandler(f_handler)

    return logger, log_path


def rescale(data, mean, std):
    return data * std + mean


if __name__ == "__main__":
    with training_session("test", "./logs", 999) as logger:
        logger.log_training_config({"lr": 1e-4, "batch_size": 32})
        logger.log_step(1, 0.5, {"acc": 0.8})
        logger.log_evaluation(1, {"mae": 0.3, "rmse": 0.4})