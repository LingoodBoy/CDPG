import sys

import numpy as np
import torch
from numpy import random
from sklearn.datasets import make_moons


def datapreparing(targetDataset, basemodel, modelparam_path=None, kgemb_path=None, timeemb_path=None):
    
    if targetDataset=='None':
        trainid = list(range(0, 656))
        genid = list(range(0, 656))
    elif targetDataset == 'sh':
        trainid = list(range(4505, 12207)) + list(range(12207, 20207))  # nc+nj
        genid = list(range(0, 4505))        # sh
    elif targetDataset == 'nc':
        trainid = list(range(0, 4505)) + list(range(12207, 20207))  # sh+nj
        genid = list(range(4505, 12207))    # nc
    elif targetDataset == 'nj':
        trainid = list(range(0, 4505)) + list(range(4505, 12207))  # sh+nc
        genid = list(range(12207, 20207))   # nj

    if modelparam_path is not None:
        print(f"Use the specified model parameter file: {modelparam_path}")
        rawdata = np.load(modelparam_path)
    elif basemodel=='v_Trans':  #
        rawdata = np.load('../data/model_params/ModelParams_v_Trans_6720.npy')
    
    genTarget = rawdata[genid]
    training_seq = rawdata[trainid]
    

    channel = 415
    repeatNum = 30
    
    training_seq = training_seq.reshape(training_seq.shape[0], channel, -1)
    genTarget = genTarget.reshape(genTarget.shape[0], channel, -1)
    scale = np.max(np.abs(training_seq))

    if scale < 1:
        scale = 1
    scale = 1

    training_seq = training_seq/scale  
    training_seq = training_seq.astype(np.float32)
    genTarget = genTarget/scale 
    genTarget = genTarget.astype(np.float32)
    if kgemb_path is not None:
        print(f"Use the specified spatial embedding file: {kgemb_path}")
        kgEmb = np.load(kgemb_path)
    else :
        # sh+nc+nj
        print(f"No specified spatial embedding file")
        kgEmb = np.ones((20207, 128))
    kgEmb = kgEmb.astype(np.float32)

    kgtrainEmb = kgEmb[trainid] 
    kggenEmb = kgEmb[genid] 

    if timeemb_path is not None:
        print(f"Use the specified temporal embedding file: {timeemb_path}")
        timeEmb = np.load(timeemb_path)

    timeEmb = timeEmb.astype(np.float32)

    timetrainEmb = timeEmb[trainid]
    timegenEmb = timeEmb[genid]
    return training_seq, scale, kgtrainEmb, kggenEmb, timetrainEmb, timegenEmb, genTarget
    
    
