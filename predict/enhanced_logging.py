#!/usr/bin/env python3

import time
from typing import Dict, Any, Optional

def enhance_pretrain_logging(logger, args: Dict[str, Any]):
    logger.info("=" * 80)
    logger.info("Pretrain")
    logger.info("=" * 80)
    config_info = [
        f"dataset: {args.get('test_dataset', 'N/A')}",
        f"mode: {args.get('taskmode', 'N/A')}",
        f"epoches: {args.get('epochs', 'N/A')}",
        f"target days: {args.get('target_days', 'N/A')}",
        f"device: {args.get('device', 'N/A')}"
    ]
    
    for info in config_info:
        logger.info(info)
    
    logger.info("=" * 80)


def log_node_training_start(logger, node_id: int, total_nodes: int, dataset: str):
    progress = (node_id + 1) / total_nodes * 100
    logger.info(f"Start training node {node_id+1:3d}/{total_nodes} ({progress:5.1f}%) | dataset: {dataset}")


def log_node_training_complete(logger, node_id: int, total_nodes: int, 
                              metrics: Dict[str, float], training_time: float):
    progress = (node_id + 1) / total_nodes * 100

    metric_strs = []
    for key, value in metrics.items():
        if isinstance(value, list):
            if len(value) <= 3:
                metric_strs.append(f"{key}: {[f'{v:.4f}' for v in value]}")
            else:
                metric_strs.append(f"{key}: [{value[0]:.4f}, {value[1]:.4f}, {value[2]:.4f}...]")
        else:
            metric_strs.append(f"{key}: {value:.4f}")
    
    metric_info = " | ".join(metric_strs)
    logger.info(f"node {node_id+1:3d}/{total_nodes} complete ({progress:5.1f}%) | cost time: {training_time:.2f}s | {metric_info}")


def log_epoch_progress(logger, epoch: int, total_epochs: int, loss: float, 
                      lr: Optional[float] = None):
    progress = (epoch + 1) / total_epochs * 100
    log_parts = [f"📈 Epoch {epoch+1:3d}/{total_epochs} ({progress:5.1f}%)", f"Loss: {loss:.6f}"]
    
    if lr is not None:
        log_parts.append(f"LR: {lr:.6f}")
    
    logger.info(" | ".join(log_parts))


def log_final_results(logger, all_results: Dict[str, Any], total_time: float):
    logger.info("=" * 80)
    logger.info("Training Complete")
    logger.info("=" * 80)

    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = total_time % 60
    logger.info(f"Total Cost time: {hours:02d}:{minutes:02d}:{seconds:05.2f}")

    for key, value in all_results.items():
        if isinstance(value, dict):
            logger.info(f"📊 {key}:")
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, list):
                    logger.info(f"    {sub_key}: {[f'{v:.4f}' for v in sub_value[:6]]}")
                else:
                    logger.info(f"    {sub_key}: {sub_value:.4f}")
        else:
            logger.info(f"{key}: {value}")
    
    logger.info("=" * 80)


def create_enhanced_progress_bar(total: int, desc: str = "Training progress"):
    from tqdm import tqdm
    
    return tqdm(
        total=total,
        desc=f"{desc}",
        ncols=120,
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}',
        ascii=False
    )