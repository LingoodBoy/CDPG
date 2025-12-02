from PredictionModel.NetSet import StgnnSet
import sys
import time
from PredictionModel.datasets import *
from PredictionModel.utils import *
from tqdm import tqdm


def getNewIndex(nodeindex, addr):
    return 0


def evalParams(params, config, device, logger, targetDataset, writer, epoch, basemodel):
    data_args, task_args, model_args = config['data'], config['task'], config['model']
    model = StgnnSet(data_args, task_args, model_args, basemodel).to(device=device)
    logger.info(f"Start parameter evaluation | Epoch: {epoch} | BaseModel: {basemodel} | Dataset: {targetDataset}")

    node_num = 0
    epochs = 0
    datasetsall = [targetDataset]

    logger.info(f"Evaluate the dataset: {datasetsall}")

    results = []
    total_evaluation_time = 0

    dataset = datasetsall[0] 
    from PredictionModel.utils import compute_nodewise_normalization_params
    dataset_path = data_args[dataset]['dataset_path']
    node_log_means, node_log_stds, normalized_data = compute_nodewise_normalization_params(dataset_path, train_ratio=0.85)
    logger.info(f"{dataset}Dataset preprocessing completed, standardize data shape: {normalized_data.shape}")

    for dataset_idx, dataset in enumerate(datasetsall):
        dataset_start_time = time.time()
        logger.info(f"Evaluate the dataset {dataset_idx+1}/{len(datasetsall)}: {dataset}")
    
        if dataset == 'sh':
            node_num = data_args[dataset]['node_num'] 
            locbias = 0
        elif dataset == 'nc':
            node_num = data_args[dataset]['node_num'] 
            locbias = 0
        elif dataset == 'nj':
            node_num = data_args[dataset]['node_num']
            locbias = 0

        logger.info(f"Dataset Configuration | Number of Nodes: {node_num} | Location Bias: {locbias}")

        if dataset in ['sh', 'nc', 'nj']:
            import random
            random.seed(42)
            eval_nodes = random.sample(range(node_num), 300)
            logger.info(f"Randomly sample 300 nodes for evaluation (fixed seed 42), selected from {node_num} nodes")
        else:
            eval_nodes = range(node_num)

        step = 0
        params = params.reshape((params.shape[0],-1))

        node_pbar = tqdm(eval_nodes,
                        desc=f"evaluate {dataset}",
                        ncols=100,
                        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}')
        
        for node_index in node_pbar:
            node_start_time = time.time()
            model = StgnnSet(data_args, task_args, model_args, model=basemodel).to(device=device)
            
            test_dataset = traffic_dataset2(data_args, task_args, node_index, "test", False, dataset,
                                           norm_params={'node_log_means': node_log_means, 'node_log_stds': node_log_stds, 'normalized_data': normalized_data})
            test_scaler = test_dataset.scaler
            test_dataloader = DataLoader(test_dataset, batch_size=task_args['test_batch_size'], shuffle=True, num_workers=0, pin_memory=False, drop_last=True)
            AAddr = data_args[dataset]['adjacency_matrix_path']
            
            newindex = getNewIndex(node_index,AAddr)
            
            param = params[node_index + locbias]
            outputs, y_label = model.task4eval(param, newindex, node_index, test_dataloader, logger, test_scaler, basemodel) 
            
            node_time = time.time() - node_start_time
            if dataset in ['sh', 'nc', 'nj']:
                node_pbar.set_postfix({
                    'node': f'{node_index:4d}',
                    'time': f'{node_time:.2f}s'
                })
            else:
                node_pbar.set_postfix({
                    'node': f'{node_index+1:3d}/{node_num}',
                    'time': f'{node_time:.2f}s'
                })
            
            if step == 0 :
                out = outputs
                truth = y_label
                step = 1
            else:
                out = np.concatenate((out, outputs),axis=2)
                truth = np.concatenate((truth, y_label),axis=2)
        
        node_pbar.close()

        result = metric_func(pred=out, y=truth, times=task_args['pred_num'])
        results.append(result)
        
        dataset_time = time.time() - dataset_start_time
        total_evaluation_time += dataset_time
        mae_mean = np.mean(result['MAE'])
        rmse_mean = np.mean(result['RMSE'])
        
        logger.info(f"{dataset} Evaluation completed | Time consumed: {dataset_time:.2f}s | MAE: {mae_mean:.4f} | RMSE: {rmse_mean:.4f}")

    metricsum = []
    logger.info(f"Calculate the final evaluation metrics...")
    
    for idx, result in enumerate(results):
        metricsum.append(result_print(result, logger, info_name=f'Dataset{idx+1}'))
    
    total_metric = sum(metricsum)
    logger.info(f"Parameter evaluation completed | Total time consumed: {total_evaluation_time:.2f}s | Metric sum: {total_metric:.6f}")
    
    return total_metric
        
        
        
        