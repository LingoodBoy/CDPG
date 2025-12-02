import copy
import random
import os
from functools import lru_cache

import numpy as np
import torch
from torch.utils.data.sampler import *
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from utils import *
from utils.train_utils import get_normalized_adj, generate_dataset

_ADJACENCY_CACHE = {}
_DATA_CACHE = {}

def load_adjacency_matrix_cached(addr):

    if addr not in _ADJACENCY_CACHE:
        _ADJACENCY_CACHE[addr] = np.load(addr, allow_pickle=True)
    return _ADJACENCY_CACHE[addr]

def load_data_cached(addr):

    if addr not in _DATA_CACHE:
        _DATA_CACHE[addr] = np.load(addr, allow_pickle=True)
    return _DATA_CACHE[addr]


def loadAM(addr, nodeindex, max_nodes=780):


    A = load_adjacency_matrix_cached(addr) 

    neighbors_mask = (A[nodeindex, :] > 0) | (A[:, nodeindex] > 0)
    neighbors = np.where(neighbors_mask)[0]

    other_neighbors = neighbors[neighbors != nodeindex].tolist()
    nodeset = [nodeindex] + other_neighbors

    if len(nodeset) > max_nodes:
        nodeset = nodeset[:max_nodes]

    n = len(nodeset)
    sm = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            sm[i][j] = A[nodeset[i]][nodeset[j]]
        # sm[i][i] = 1
    return sm, nodeset, A

class BBDefinedError(Exception):
    def __init__(self,ErrorInfo):
        super().__init__(self) 
        self.errorinfo=ErrorInfo
    def __str__(self):
        return self.errorinfo


class traffic_dataset2(Dataset):
    def __init__(self, data_args, task_args, nodeindex, stage='source', ifchosenode=False, test_data='metr-la', add_target=True, target_days=3, ifspatial=1,  datanum=0.7, norm_params=None):
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
        self.mean = 0
        self.std = 0
        self.datanum = datanum
        self.norm_params = norm_params 
        self.load_data(stage, ifchosenode, test_data, nodeindex, ifspatial)
        if self.add_target:
            self.data_list = np.append(self.data_list, self.test_data)
        

    def load_data(self, stage, ifchosenode, test_data, nodeindex, ifspatial):  
        self.A_list, self.edge_index_list = {}, {}
        self.edge_attr_list, self.node_feature_list = {}, {}
        self.x_list, self.y_list = {}, {}
        self.means_list, self.stds_list = {}, {}

        data_keys = np.array(self.data_args['data_keys'])
        if stage == 'source':
            self.data_list = np.delete(data_keys, np.where(data_keys == test_data))
        elif stage == 'singlePretrain' or stage == 'test' or stage == 'target': 
            self.data_list = np.array([test_data])
        else:
            raise BBDefinedError('Error: Unsupported Stage')
        for dataset_name in self.data_list:
            if ifchosenode==True:
                AAddr = self.data_args[dataset_name]['adjacency_matrix_path']
            else:
                AAddr = self.data_args[dataset_name].get('iadjacency_matrix_path', self.data_args[dataset_name]['adjacency_matrix_path'])
            A, nodeset, initA = loadAM(AAddr, nodeindex, max_nodes=780)
            edge_index, edge_attr, node_feature = self.get_attr_func(A)

            self.A_list[dataset_name] = torch.from_numpy(get_normalized_adj(A))
            self.edge_index_list[dataset_name] = edge_index
            self.edge_attr_list[dataset_name] = edge_attr
            self.node_feature_list[dataset_name] = node_feature

            if dataset_name=='BM' or dataset_name=='DC' or dataset_name=='man' or dataset_name=='bj':
                XAddr = self.data_args[dataset_name]['dataset_path']
            else:
                if ifchosenode==True:
                    XAddr = self.data_args[dataset_name]['dataset_path']
                else:
                    XAddr = self.data_args[dataset_name].get('idataset_path', self.data_args[dataset_name]['dataset_path'])

            X = load_data_cached(XAddr) 
            X = X[:,nodeset,:]
            X = X.transpose((1, 2, 0))
            X = X.astype(np.float32)

            if stage == 'source' or stage == 'singlePretrain':
                X = X[:, :, :int(X.shape[2]*0.85)]
            elif stage == 'target':
                print('self.target_days: ', self.target_days)
                if dataset_name in ['sh', 'nj', 'nc']:
                    steps_per_day = 48 
                else:
                    steps_per_day = 288  
                print(f'dataset: {dataset_name}, steps_per_day: {steps_per_day}, target_days: {self.target_days}')

                total_steps = X.shape[2]
                test_start_idx = int(total_steps * 0.85) 
                train_steps = self.target_days * steps_per_day 
                train_start_idx = test_start_idx - train_steps

                if train_start_idx < 0:
                    print(f'target_days={self.target_days}-{train_steps}step-testing set-{test_start_idx}')
                    train_start_idx = 0

                X = X[:, :, train_start_idx:test_start_idx] 
                print(f'range: [{train_start_idx}, {test_start_idx}), total: {test_start_idx-train_start_idx}step = {(test_start_idx-train_start_idx)/steps_per_day:.2f}day')
            elif stage == 'test':
                X = X[:, :, int(X.shape[2]*0.85):] 
            else:
                raise BBDefinedError('Error: Unsupported Stage')

            if self.norm_params is not None:
                node_log_means = self.norm_params['node_log_means']
                node_log_stds = self.norm_params['node_log_stds']

                n_nodes, n_features, n_timesteps = X.shape
                for i in range(n_nodes):
                    for f in range(n_features):
                        X[i, f, :] = np.log10(X[i, f, :]+1)
                        X[i, f, :] = (X[i, f, :] - node_log_means[i, f]) / node_log_stds[i, f]

                self.node_log_means = node_log_means
                self.node_log_stds = node_log_stds
                self.mean = node_log_means[0, 0]
                self.std = node_log_stds[0, 0]

            else:
                n_nodes, n_features, n_timesteps = X.shape
                X = np.log10(X+1)
                node_log_means = np.zeros((n_nodes, n_features))
                node_log_stds = np.zeros((n_nodes, n_features))

                for i in range(n_nodes):
                    for f in range(n_features):
                        node_data = X[i, f, :]
                        node_mean = np.mean(node_data)
                        node_std = np.std(node_data)

                        if node_std < 1e-8:
                            node_std = 1.0

                        X[i, f, :] = (node_data - node_mean) / node_std
                        node_log_means[i, f] = node_mean
                        node_log_stds[i, f] = node_std

                self.node_log_means = node_log_means
                self.node_log_stds = node_log_stds
                self.mean = node_log_means[0, 0]
                self.std = node_log_stds[0, 0]
                self.computed_norm_params = {
                    'node_log_means': node_log_means,
                    'node_log_stds': node_log_stds
                }

            dummy_means = np.array([self.mean])
            dummy_stds = np.array([self.std])
            x_inputs, y_outputs = generate_dataset(X, self.task_args['his_num'], self.task_args['pred_num'], dummy_means, dummy_stds)
            
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
    
    def node_inverse_transform(self, normalized_data, node_index, feature_index=0):
        node_log_mean = self.node_log_means[node_index, feature_index]
        node_log_std = self.node_log_stds[node_index, feature_index]
        log_scale_data = normalized_data * node_log_std + node_log_mean
        
        return log_scale_data
    
    def batch_inverse_transform(self, normalized_data, node_indices):
        denormalized_data = normalized_data.copy()
        for i, node_idx in enumerate(node_indices):
            node_log_mean = self.node_log_means[node_idx, 0]
            node_log_std = self.node_log_stds[node_idx, 0]
            denormalized_data[:, i, :] = normalized_data[:, i, :] * node_log_std + node_log_mean
            
        return denormalized_data

    def __getitem__(self, index):
        if self.stage == 'source':
            random.seed(42)
            select_dataset = random.choice(self.data_list) 
            batch_size = self.task_args['batch_size']
            torch.manual_seed(42)
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
        random.seed(42)
        select_dataset = random.choice(self.data_list)
        batch_size = self.task_args['batch_size']

        for i in range(task_num * 2):
            torch.manual_seed(42 + i)
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
            return 100000000
        else:
            data_length = self.x_list[self.data_list[0]].shape[0]
            return data_length
        
    def len(self):
        pass


    def get(self):
        pass



