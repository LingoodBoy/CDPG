# -*- coding: utf-8 -*-
import argparse
import os
import sys
import time
from datetime import datetime
from multiprocessing import cpu_count

import numpy as np
import torch
import yaml
import random

from accelerate import Accelerator


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed has been set to: {seed}")

from datapreparing import datapreparing 
from parameter_diffusion import GaussianDiffusion1D, Trainer1D 
from diffusionutils import * 
from logger_utils import DiffusionTrainingLogger, training_session 
from numpy import random

from conditional_denoiser import TransformerDenoiser
from conditional_denoiser.utils import XEDataset

from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
torch.set_num_threads(20)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameter generation script')

    parser.add_argument("--modeldim", type=int, default=256, 
                       help="Latent dimension size of the Transformer model")
    parser.add_argument("--epochs", type=int, default=100001, 
                       help="Total number of training epochs")
    parser.add_argument("--expIndex", type=int, default=888, 
                       help="Experiment index")
    parser.add_argument("--targetDataset", type=str, default='None', 
                       help="Target dataset name")
    parser.add_argument("--diffusionstep", type=int, default=500,
                       help="Number of timesteps in the diffusion process")
    parser.add_argument("--basemodel", type=str, default='v_Trans', 
                       help="Base model type")
    parser.add_argument("--batchsize", type=int, default=64,
                       help="Batch size")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--modelparam", type=str, default=None,
                       help="Path to the model parameter file")
    parser.add_argument("--spatial", type=str, default=None,
                       help="Path to spatial embedding file")
    parser.add_argument("--temporal", type=str, default=None,
                       help="Path to temporal embedding file")
 
    args = parser.parse_args()

    set_seed(args.seed)

    with training_session(
        name="GPD_Diffusion",
        log_dir="./Logs",
        exp_index=args.expIndex,
        enable_tensorboard=True,
        console_level="INFO",
        file_level="DEBUG"
    ) as enhanced_logger:

        train_lr = 0.001

        training_config = {
            "experiment_index": args.expIndex,
            "target_dataset": args.targetDataset,
            "base_model": args.basemodel,
            "diffusion_steps": args.diffusionstep,
            "epochs": args.epochs,
            "batch_size": args.batchsize,
            "model_dim": args.modeldim,
            "learning_rate": train_lr,
            "gradient_accumulate_every": 1,
            "save_and_sample_every": 3000,
            "ema_decay": 0.995,
            "random_seed": args.seed
        }
        enhanced_logger.log_training_config(training_config)

        writer = enhanced_logger.tb_writer
        logger = enhanced_logger.logger

        training_seq, scale, kgtrainEmb, kggenEmb, timetrainEmb,timegenEmb, genTarget = datapreparing(
            args.targetDataset, args.basemodel, args.modelparam, args.spatial, args.temporal
            )  

        data_info = {
            'training_seq_shape': training_seq.shape,
            'region_embedding_shape': kgtrainEmb.shape,
            'time_embedding_shape': timetrainEmb.shape,
            'scale': scale
        }

        for key, value in data_info.items():
            enhanced_logger.info(f"  {key}: {value}")

        logger.info('training_seq.shape: {}'.format(training_seq.shape))
        logger.info('regionEmbedding.shape: {}'.format(kgtrainEmb.shape))
        logger.info('timeEmbedding.shape: {}'.format(timetrainEmb.shape))
        logger.info('Training hyperparameters: {}'.format(vars(args)))

        XEData = XEDataset(training_seq, kgtrainEmb, timetrainEmb)
        enhanced_logger.info(f"Dataset created, number of samples: {len(XEData)}")

        experimentIndex = args.expIndex 
        deffusionStep = args.diffusionstep 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    numEpoch = args.epochs 
    batchsize = args.batchsize    
    sampleTimes = 1  
    transformer_dim = args.modeldim 

    d_model = transformer_dim 
    q = 10 
    v = 10 
    h = 10 
    N = 4 
    d_kgEmb = kgtrainEmb.shape[1]
    d_timeEmb = timetrainEmb.shape[1]
    attention_size = 5 
    dropout = 0.1
    pe = 'original'
    chunk_mode = None 
    d_input = training_seq.shape[1]
    d_output = d_input  
    layernum = training_seq.shape[2]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dinoisingModel = TransformerDenoiser(d_input, d_model, d_output, d_kgEmb, d_timeEmb, q, v, h, N,
                                        attention_size=attention_size, layernum=layernum,
                                        dropout=dropout, chunk_mode=chunk_mode, pe=pe).to(device)
    
    diffusion = GaussianDiffusion1D(
        dinoisingModel,  # Transformer 
        seq_length = training_seq.shape[2],
        timesteps = deffusionStep,  
        loss_type='l2',
        objective = 'pred_v',
        auto_normalize = False, 
        beta_schedule='linear',
    ).to(device) 

    outputpath = './Output/exp{}'.format(args.expIndex)
    if not os.path.exists(outputpath):
        os.makedirs(outputpath)

    modelsavepath = './ModelSave/exp{}'.format(args.expIndex)
    if not os.path.exists(modelsavepath):
        os.makedirs(modelsavepath)

    trainer = Trainer1D(
        diffusion,
        dataset = XEData,
        train_batch_size = batchsize, 
        train_lr = train_lr,
        train_num_steps = numEpoch,
        gradient_accumulate_every = 1,
        save_and_sample_every = 3000,
        results_folder = modelsavepath, 
        ema_decay = 0.995,
        amp = False, 
        logger = logger, 
        kgEmb = kggenEmb, 
        timeEmb = timegenEmb,
        genTarget = genTarget, 
        targetDataset = args.targetDataset,
        scale = scale,
        tbwriter = writer,
        outputpath = outputpath,
        sampleTimes = sampleTimes, 
        basemodel = args.basemodel,
    )

    logger.info('Training started: total epochs: {}, batch size: {}, learning rate: {}'.format(numEpoch, batchsize, train_lr))  # Log key training parameters
    trainer.train()
    if hasattr(trainer, 'best_model_file') and os.path.exists(trainer.best_model_file):
        import re
        match = re.search(r'model-(\d+)\.pt', trainer.best_model_file)
        if match:
            best_step = int(match.group(1))
            logger.info(f"Load the best model for final sampling | Best Step: {best_step} | file: {trainer.best_model_file}")
            trainer.load(best_step)

    kggenEmb = torch.tensor(kggenEmb).to(device)
    timegenEmb = torch.tensor(timegenEmb).to(device)
    sampleRes = None
    for _ in range(sampleTimes):
        if args.targetDataset == 'sh':
            length = len(kggenEmb)
            startNum = [0, length // 5, 2 * (length // 5), 3 * (length // 5), 4 * (length // 5), length]
            generateNum = np.diff(startNum)
            result = None

            for id in range(len(generateNum)):
                smallkgEmb = kggenEmb[startNum[id]:startNum[id+1]]
                smalltimeEmb = timegenEmb[startNum[id]:startNum[id+1]]
                sampled_seq = diffusion.sample(
                    torch.tensor(smallkgEmb).to(device),
                    torch.tensor(smalltimeEmb).to(device),
                    generateNum[id]
                )
                if result is None:
                    result = sampled_seq.detach().cpu()
                else:
                    result = np.concatenate((result, sampled_seq.detach().cpu()),axis=0)

        elif args.targetDataset == 'nc':
            length = len(kggenEmb)
            startNum = [0, length // 3, 2 * (length // 3), length]
            generateNum = np.diff(startNum)
            result = None

            for id in range(len(generateNum)):
                smallkgEmb = kggenEmb[startNum[id]:startNum[id+1]]
                smalltimeEmb = timegenEmb[startNum[id]:startNum[id+1]]
                sampled_seq = diffusion.sample(
                    torch.tensor(smallkgEmb).to(device),
                    torch.tensor(smalltimeEmb).to(device),
                    generateNum[id]
                )
                if result is None:
                    result = sampled_seq.detach().cpu()
                else:
                    result = np.concatenate((result, sampled_seq.detach().cpu()),axis=0)

        elif args.targetDataset == 'nj':
            length = len(kggenEmb)
            startNum = [0, length // 3, 2 * (length // 3), length]
            generateNum = np.diff(startNum)
            result = None

            for id in range(len(generateNum)):
                smallkgEmb = kggenEmb[startNum[id]:startNum[id+1]]
                smalltimeEmb = timegenEmb[startNum[id]:startNum[id+1]]
                sampled_seq = diffusion.sample(
                    torch.tensor(smallkgEmb).to(device),
                    torch.tensor(smalltimeEmb).to(device),
                    generateNum[id]
                )
                if result is None:
                    result = sampled_seq.detach().cpu()
                else:
                    result = np.concatenate((result, sampled_seq.detach().cpu()),axis=0)

        if sampleRes is None:
            sampleRes = np.expand_dims(result, axis=0)
            logger.info('Shape of the first sampling result: {}'.format(sampleRes.shape))
        else:
            sampleRes = np.concatenate((np.expand_dims(result, axis=0), sampleRes),axis=0)
            logger.info('Shape of the accumulated sampling results: {}'.format(sampleRes.shape))

    sampleRes = np.average(sampleRes, axis = 0) 
    logger.info('Shape of the final sampling result: {}'.format(sampleRes.shape)) 
    np.save(os.path.join(outputpath, 'sampleSeq_RealParams_{}.npy'.format(experimentIndex)), sampleRes)
    logger.info('Training and sample complete，results have been saving to: {}/sampleSeq_RealParams_{}.npy'.format(outputpath, experimentIndex)) 