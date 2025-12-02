import sys
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
import numpy as np
from PredictionModel.Models.graph_transformer import v_GraphTransformer
from PredictionModel.utils import *
from copy import deepcopy
from tqdm import tqdm
import scipy.sparse as sp

def asym_adj(adj):
    adj = adj.cpu().numpy()
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat= sp.diags(d_inv)
    return d_mat.dot(adj).astype(np.float32).todense()

class StgnnSet(nn.Module): 
    def __init__(self, data_args, task_args, model_args, model='', node_num=207):
        super(StgnnSet, self).__init__()
        self.data_args = data_args
        self.task_args = task_args
        self.model_args = model_args
        
        self.update_lr = model_args['update_lr']
        self.meta_lr = model_args['meta_lr']
        self.update_step = model_args['update_step']
        self.update_step_test = model_args['update_step_test']
        self.task_num = task_args['task_num']
        self.model_name = model

        self.loss_lambda = model_args['loss_lambda']
        # print("loss_lambda = ", self.loss_lambda)

        self.model = v_GraphTransformer(model_args, task_args)

        self.meta_optim = optim.Adam(self.model.parameters(), lr=self.meta_lr, weight_decay=1e-2)
        self.loss_criterion = nn.MSELoss()


    def graph_reconstruction_loss(self, meta_graph, adj_graph):   
        adj_graph = adj_graph.unsqueeze(0).float()
        for i in range(meta_graph.shape[0]):
            if i == 0:
                matrix = adj_graph
            else:
                matrix = torch.cat((matrix, adj_graph), 0)
        criteria = nn.MSELoss()
        loss = criteria(meta_graph, matrix.float())
        return loss
      
    def calculate_loss(self, out, y, meta_graph, matrix, stage='target', graph_loss=True, loss_lambda=1):
        if loss_lambda == 0:
            loss = self.loss_criterion(out, y)
        if graph_loss:
            if stage == 'source' or stage == 'target_maml':
                loss_predict = self.loss_criterion(out, y)
                loss_reconsturct = self.graph_reconstruction_loss(meta_graph, matrix)
            else:
                loss_predict = self.loss_criterion(out, y)
                loss_reconsturct = self.loss_criterion(meta_graph, matrix.float())
            loss = loss_predict + loss_lambda * loss_reconsturct
        else:
            loss = self.loss_criterion(out, y)

        return loss

    def forward(self, data, matrix):
        out, meta_graph = self.model(data, matrix)
        return out, meta_graph

    def finetuning(self, target_dataloader, test_dataloader, target_epochs,logger):
        maml_model = deepcopy(self.model)

        optimizer = optim.Adam(maml_model.parameters(), lr=self.meta_lr, weight_decay=1e-2)
        min_MAE = 10000000
        best_result = ''
        best_meta_graph = -1

        for epoch in tqdm(range(target_epochs)):
            train_losses = []
            start_time = time.time()
            maml_model.train()

            for step, (data, A_wave) in enumerate(target_dataloader):
                data, A_wave = data.cuda(), A_wave.cuda()
                data.node_num = data.node_num[0]

                batch_size, node_num, seq_len, _ = data.x.shape

                out, meta_graph = maml_model(data, A_wave[0].float())
                loss = self.calculate_loss(out, data.y, meta_graph, A_wave, 'test', loss_lambda=self.loss_lambda)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_losses.append(loss.detach().cpu().numpy())
            avg_train_loss = sum(train_losses)/len(train_losses)
            end_time = time.time()
            if epoch % 5 == 0:
                logger.info("[Target Fine-tune] epoch #{}/{}: loss is {}, fine-tuning time is {}".format(epoch+1, target_epochs, avg_train_loss, end_time-start_time))

        with torch.no_grad():
            test_start = time.time()  
            maml_model.eval()
            for step, (data, A_wave) in enumerate(test_dataloader):
                data, A_wave = data.cuda(), A_wave.cuda()
                data.node_num = data.node_num[0]
                batch_size, node_num, seq_len, _ = data.x.shape
                out, meta_graph = maml_model(data, A_wave[0].float())

                if step == 0:
                    outputs = out
                    y_label = data.y
                else:
                    outputs = torch.cat((outputs, out))
                    y_label = torch.cat((y_label, data.y))
            outputs = outputs.permute(0, 2, 1).detach().cpu().numpy()
            y_label = y_label.permute(0, 2, 1).detach().cpu().numpy()
            result = metric_func(pred=outputs, y=y_label, times=self.task_args['pred_num'])
            test_end = time.time()

            result_print(result, logger,info_name='Evaluate')
            logger.info("[Target Test] testing time is {}".format(test_end-test_start))

    def taskTrain(self, taskmode, target_dataloader, test_dataloader, target_epochs,logger):

        optimizer = optim.Adam(self.model.parameters(), lr=self.meta_lr, weight_decay=1e-2)
        min_MAE = 10000000
        best_result = ''
        best_meta_graph = -1

        for epoch in tqdm(range(target_epochs)):
            train_losses = []
            start_time = time.time()
            self.model.train()

            for step, (data, A_wave) in enumerate(target_dataloader):
                data, A_wave = data.cuda(), A_wave.cuda()
                data.node_num = data.node_num[0]

                batch_size, node_num, seq_len, _ = data.x.shape

                out, meta_graph = self.model(data, A_wave[0].float())
                loss = self.calculate_loss(out, data.y, meta_graph, A_wave, 'test', loss_lambda=self.loss_lambda)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_losses.append(loss.detach().cpu().numpy())
            avg_train_loss = sum(train_losses)/len(train_losses)
            end_time = time.time()
            if epoch % 5 == 0:
                logger.info("[Target Fine-tune] epoch #{}/{}: loss is {}, fine-tuning time is {}".format(epoch+1, target_epochs, avg_train_loss, end_time-start_time))
                if taskmode=='task3':
                    torch.save(self.model,'Param/{}_inside.pt'.format(taskmode))
        
        torch.save(self.model,'Param/{}_inside.pt'.format(taskmode))

        with torch.no_grad():
            test_start = time.time()  
            self.model.eval()
            for step, (data, A_wave) in enumerate(test_dataloader):
                data, A_wave = data.cuda(), A_wave.cuda()
                data.node_num = data.node_num[0]
                batch_size, node_num, seq_len, _ = data.x.shape
                out, meta_graph = self.model(data, A_wave[0].float())

                if step == 0:
                    outputs = out
                    y_label = data.y
                else:
                    outputs = torch.cat((outputs, out))
                    y_label = torch.cat((y_label, data.y))
            outputs = outputs.permute(0, 2, 1).detach().cpu().numpy()
            y_label = y_label.permute(0, 2, 1).detach().cpu().numpy()
            result = metric_func(pred=outputs, y=y_label, times=self.task_args['pred_num'])
            test_end = time.time()

            result_print(result, logger,info_name='Evaluate')
            logger.info("[Target Test] testing time is {}".format(test_end-test_start))
    
    
    def node_eval(self, node_index, test_dataloader, logger, test_dataset):

        with torch.no_grad():
            self.model = torch.load('Param/Task3_1/{}/task3_{}.pt' .format(test_dataset, node_index))
            test_start = time.time()  
            self.model.eval()
            for step, (data, A_wave) in enumerate(test_dataloader):

                A_wave = A_wave[0][node_index,:].unsqueeze(0)
                data, A_wave = data.cuda(), A_wave.cuda()
                data.node_num = data.node_num[0]
                batch_size, node_num, seq_len, _ = data.x.shape

                out, meta_graph = self.model(data, A_wave.float())

                if step == 0:
                    outputs = out
                    y_label = data.y
                else:
                    outputs = torch.cat((outputs, out))
                    y_label = torch.cat((y_label, data.y))
            outputs = outputs.permute(0, 2, 1).detach().cpu().numpy()
            y_label = y_label.permute(0, 2, 1).detach().cpu().numpy()
            result = metric_func(pred=outputs, y=y_label, times=self.task_args['pred_num'])
            test_end = time.time()

            result_print(result, logger,info_name='Evaluate')
            logger.info("[Test] testing time is {}".format(test_end-test_start))
            return outputs, y_label
    
    def taskEval(self, test_dataloader, logger):
        self.model = torch.load('Param/task1.pt')
        with torch.no_grad():
            test_start = time.time()  
            self.model.eval()
            for step, (data, A_wave) in enumerate(test_dataloader):
                data, A_wave = data.cuda(), A_wave.cuda()
                data.node_num = data.node_num[0]
                batch_size, node_num, seq_len, _ = data.x.shape

                out, meta_graph = self.model(data, A_wave[0].float())

                if step == 0:
                    outputs = out
                    y_label = data.y
                else:
                    outputs = torch.cat((outputs, out))
                    y_label = torch.cat((y_label, data.y))
            outputs = outputs.permute(0, 2, 1).detach().cpu().numpy()
            y_label = y_label.permute(0, 2, 1).detach().cpu().numpy()
            result = metric_func(pred=outputs, y=y_label, times=self.task_args['pred_num'])
            test_end = time.time()

            result_print(result, logger,info_name='Evaluate')
            logger.info("[Target Test] testing time is {}".format(test_end-test_start))

    def task4eval(self, param, node_index, init_index, test_dataloader, logger, test_scaler, basemodel):
        '''eval the sample from diffusion '''

        indexstart = [
            0, 192, 208, 464, 480, 736, 752, 1008,
            1024, 1280, 1296, 1552, 1568, 1824, 1840, 2096,
            2112, 2368, 2384, 2400, 2416, 2432, 2448, 2464,
            2480, 2992, 3024, 3536, 3552, 6624, 6640,
        ]
        shapes = [
            (12, 16), (16, 1), (16, 16), (16,),
            (16, 16), (16,), (16, 16), (16,),
            (16, 16), (16,), (16, 16), (16,),
            (16, 16), (16,), (16, 16), (16,),
            (16, 16), (16,), (16,), (16,),
            (16,), (16,), (16,), (16,),
            (32, 16), (32,), (16, 32), (16,),
            (16, 192), (1, 16),
        ]
            
        '''load model'''
        with torch.no_grad():
            # self.model = torch.load('Param/Task4/{}/task4_{}.pt' .format(test_dataset, init_index))
            index = 0
            for key in self.model.state_dict().keys():
                if ('running_mean' not in key) and ('running_var' not in key) and ('num_batches_tracked' not in key) and ('pe' not in key):
                    pa = torch.tensor(param[indexstart[index]:indexstart[index+1]])
                    pa = torch.reshape(pa, shapes[index])                
                    self.model.state_dict()[key].copy_(pa)               
                    index = index+1
            # print('params load ok')
            # sys.exit(0)

            self.model.eval()

            outputs = None
            y_label = None

            for step, (data, A_wave) in enumerate(test_dataloader):

                data, A_wave = data.cuda(), A_wave.cuda()
                data.node_num = data.node_num[0]

                out, meta_graph = self.model(data, A_wave[0].float())

                out_cpu = out[:,node_index,:].detach().cpu().numpy()  # (batch, time_steps)
                datay_cpu = data.y[:,node_index,:].detach().cpu().numpy()  # (batch, time_steps)

                node_log_mean = test_scaler.node_log_means[node_index] 
                node_log_std = test_scaler.node_log_stds[node_index] 

                out_denorm = out_cpu * node_log_std + node_log_mean      # (batch, time_steps)
                datay_denorm = datay_cpu * node_log_std + node_log_mean  # (batch, time_steps)

                out = torch.from_numpy(out_denorm).to(out.device)
                datay = torch.from_numpy(datay_denorm).to(data.y.device)

                if step == 0:
                    outputs = out.unsqueeze(1)  
                    y_label = datay.unsqueeze(1)
                else:
                    outputs = torch.cat((outputs, out.unsqueeze(1)),dim=1)
                    y_label = torch.cat((y_label, datay.unsqueeze(1)),dim=1)         

            outputs = outputs.permute(0, 2, 1).detach().cpu().numpy()
            y_label = y_label.permute(0, 2, 1).detach().cpu().numpy()     

            return outputs, y_label
