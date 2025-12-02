import copy
import random

import numpy as np
import torch
from PredictionModel.utils import *
from torch.utils.data.sampler import *
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from sklearn.preprocessing import StandardScaler


def loadAM(addr, nodeindex):
    A = np.load(addr)
    nodeset = []
    for i in range(A.shape[0]):
        if A[i][nodeindex] != 0  or A[nodeindex][i] != 0:
            nodeset.append(i)

    n = len(nodeset)
    sm = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            sm[i][j] = A[nodeset[i]][nodeset[j]]

    return sm, nodeset, A

class BBDefinedError(Exception):
    def __init__(self,ErrorInfo):
        super().__init__(self) 
        self.errorinfo=ErrorInfo
    def __str__(self):
        return self.errorinfo

class traffic_dataset2(Dataset):
    def __init__(self, data_args, task_args, nodeindex, stage='source', ifchosenode=False, test_data='sh', add_target=True, target_days=3, ifspatial=1, norm_params=None):
        super(traffic_dataset2, self).__init__()
        self.data_args = data_args
        self.task_args = task_args
        self.his_num = task_args['his_num']
        self.pred_num = task_args['pred_num']
        self.stage = stage
        self.add_target = add_target
        self.test_data = test_data
        self.target_days = target_days
        self.nodeindex = nodeindex
        self.norm_params = norm_params
        self.load_data(stage, ifchosenode, test_data, nodeindex, ifspatial)
        if self.add_target:
            self.data_list = np.append(self.data_list, self.test_data)
        # print("[INFO] Dataset init finished!")

    def load_data(self, stage, ifchosenode, test_data, nodeindex, ifspatial):   
        self.A_list, self.edge_index_list = {}, {}
        self.edge_attr_list, self.node_feature_list = {}, {}
        self.x_list, self.y_list = {}, {}
        self.means_list, self.stds_list = {}, {}

        data_keys = np.array(self.data_args['data_keys'])
        if stage == 'source':
            self.data_list = np.delete(data_keys, np.where(data_keys == test_data))
        elif stage == 'task1':
            self.data_list = data_keys  
        elif stage == 'target' or stage == 'target_maml' or stage == 'task2': 
            self.data_list = np.array([test_data])
        elif stage == 'test':
            self.data_list = np.array([test_data])
        elif stage == 'dann':
            self.data_list = np.array([test_data])
        else:
            raise BBDefinedError('Error: Unsupported Stage')

        for dataset_name in self.data_list:
            AAddr = self.data_args[dataset_name]['adjacency_matrix_path'] 
            
            A, nodeset, initA = loadAM(AAddr, nodeindex)
            edge_index, edge_attr, node_feature = self.get_attr_func(A)

            self.A_list[dataset_name] = torch.from_numpy(get_normalized_adj(A))
            # initA2 = torch.from_numpy(get_normalized_adj(initA))
            self.edge_index_list[dataset_name] = edge_index
            self.edge_attr_list[dataset_name] = edge_attr
            self.node_feature_list[dataset_name] = node_feature

            if dataset_name=='BM' or dataset_name=='DC' or dataset_name=='man' or dataset_name=='bj':
                XAddr = self.data_args[dataset_name]['dataset_path']
            else:
                XAddr = self.data_args[dataset_name]['dataset_path']

            if self.norm_params is None or 'normalized_data' not in self.norm_params:
                raise ValueError("Normalization data must be provided.")

            global_normalized_data = self.norm_params['normalized_data'] 
            node_log_means = self.norm_params['node_log_means'] 
            node_log_stds = self.norm_params['node_log_stds'] 

            normalized_data_subset = global_normalized_data[:, nodeset]

            class NodewiseScaler:
                def __init__(self, node_log_means, node_log_stds, nodeset):
                    self.node_log_means = node_log_means[nodeset] 
                    self.node_log_stds = node_log_stds[nodeset]   
                    self.nodeset = nodeset

            self.scaler = NodewiseScaler(node_log_means, node_log_stds, nodeset)

            normalized_data_subset = normalized_data_subset[..., np.newaxis]
            X = normalized_data_subset.transpose((1, 2, 0)) 
            X = X.astype(np.float32)  

            if stage == 'source' or stage == 'dann':
                X = X
            elif stage == 'task1':
                X = X[:, :, int(X.shape[2]*0.15):]
            elif stage =='task2':
                X = X[:, :, int(X.shape[2]*0.15):]
            elif stage == 'target' or stage == 'target_maml':
                X = X[:, :, :288 * self.target_days]
            elif stage == 'test':
                X = X[:, :, int(X.shape[2]*0.85):]  
            else:
                raise BBDefinedError('Error: Unsupported Stage')


            x_inputs, y_outputs = generate_dataset(X, self.task_args['his_num'], self.task_args['pred_num'])

            self.x_list[dataset_name] = x_inputs
            self.y_list[dataset_name] = y_outputs

    def get_attr_func(self, matrix, edge_feature_matrix_path=None, node_feature_path=None):
        a, b = [], []
        edge_attr = []
        node_feature = None
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if(matrix[i][j] > 0):
                    a.append(i)
                    b.append(j)
        edge = [a,b]
        edge_index = torch.tensor(edge, dtype=torch.long)

        return edge_index, edge_attr, node_feature
    
    def get_edge_feature(self, edge_index, x_data):
        pass

    def __getitem__(self, index):

        if self.stage == 'source':
            select_dataset = random.choice(self.data_list) 
            batch_size = self.task_args['batch_size']
            permutation = torch.randperm(self.x_list[select_dataset].shape[0])
            indices = permutation[0: batch_size]
            x_data = self.x_list[select_dataset][indices]
            y_data = self.y_list[select_dataset][indices]
        
        elif self.stage == 'target_maml':
            select_dataset = self.data_list[0]
            batch_size = self.task_args['batch_size']
            permutation = torch.randperm(self.x_list[select_dataset].shape[0])
            indices = permutation[0: batch_size]
            x_data = self.x_list[select_dataset][indices]
            y_data = self.y_list[select_dataset][indices]

        else:
            select_dataset = self.data_list[0]
            x_data = self.x_list[select_dataset][index: index+1]
            y_data = self.y_list[select_dataset][index: index+1]

        node_num = self.A_list[select_dataset].shape[0]
        data_i = Data(node_num=node_num, x=x_data, y=y_data)
        data_i.edge_index = self.edge_index_list[select_dataset]
        data_i.data_name = select_dataset
        A_wave = self.A_list[select_dataset]
        return data_i, A_wave
    
    def get_maml_task_batch(self, task_num):  
        spt_task_data, qry_task_data = [], []
        spt_task_A_wave, qry_task_A_wave = [], []

        select_dataset = random.choice(self.data_list)
        batch_size = self.task_args['batch_size']

        for i in range(task_num * 2):
            permutation = torch.randperm(self.x_list[select_dataset].shape[0])
            indices = permutation[0: batch_size]
            x_data = self.x_list[select_dataset][indices]
            y_data = self.y_list[select_dataset][indices]
            node_num = self.A_list[select_dataset].shape[0]
            data_i = Data(node_num=node_num, x=x_data, y=y_data)
            data_i.edge_index = self.edge_index_list[select_dataset]
            data_i.data_name = select_dataset
            A_wave = self.A_list[select_dataset].float()

            if i % 2 == 0:
                spt_task_data.append(data_i.cuda())
                spt_task_A_wave.append(A_wave.cuda())
            else:
                qry_task_data.append(data_i.cuda())
                qry_task_A_wave.append(A_wave.cuda())

        return spt_task_data, spt_task_A_wave, qry_task_data, qry_task_A_wave

    
    def __len__(self):
        if self.stage == 'source':
            print("[random permutation] length is decided by training epochs")
            return 100000000
        else:
            data_length = self.x_list[self.data_list[0]].shape[0]
            return data_length
        
    def len(self):
        pass


    def get(self):
        pass
    
    