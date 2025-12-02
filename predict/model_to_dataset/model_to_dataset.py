import sys
sys.path.append('../../Pretrain')

import argparse
import time 

import torch  
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim 
import yaml 
import numpy as np 

from datasets import *  
from NetSet import *         
from torch.utils.data.sampler import * 
from torch_geometric.loader import DataLoader 
from tqdm import tqdm      
from utils import *    

if __name__ == '__main__':

    with open('../config.yaml') as f:
        config = yaml.full_load(f)
    data_args, task_args, model_args = config['data'], config['task'], config['model']

    model = StgnnSet(data_args, task_args, model_args, model='v_Trans')

    ifFirstRegion = True

    for dataset in ['sh', 'nc', 'nj']:
        if dataset == 'sh':     
            nodenum = 4505     
        elif dataset == 'nc':     
            nodenum = 7702      
        elif dataset == 'nj':     
            nodenum = 8000      
        for nodeindex in tqdm(range(nodenum)):

            lengthlist = []  
            startlist  = [0] 
            shapelist  = [] 

            model.model = torch.load('../Param/pretrain/{}_v_Trans_0-{}_traffic_pretrain_10.9/node_{}.pt'.format(dataset,nodenum,nodeindex), map_location=torch.device('cpu'))

            allparams = list(model.model.named_parameters())

            iffirst = True

            for singleparams in allparams:

                astensor = singleparams[1].clone().detach() 

                shapelist.append(astensor.shape)

                tensor1D = astensor.flatten()

                lengthlist.append(tensor1D.shape[0])

                tensor1D = tensor1D.unsqueeze(0)

                if iffirst == True:

                    finaltensor = tensor1D
                    iffirst = False
                else:

                    finaltensor = torch.cat((finaltensor,tensor1D), dim = 1)

                startlist.append(finaltensor.shape[1])

            if ifFirstRegion==True:

                allRegionTensor = finaltensor
                ifFirstRegion = False
            else: 

                allRegionTensor = torch.cat((allRegionTensor, finaltensor), dim=0)

    np.save('ModelParams_v_Trans_all_6720_10.9.npy', allRegionTensor.cpu())  

    print("Shape of the final parameter matrix:", allRegionTensor.shape)
