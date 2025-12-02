import os
import zipfile
import numpy as np
import torch
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
import sys
import logging
import xlwt
import xlrd
import sys

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def compute_nodewise_normalization_params(dataset_path, train_ratio=0.85):

    raw_data = np.load(dataset_path)

    if raw_data.ndim == 3:
        raw_data = raw_data.squeeze(-1)

    log_data = np.log10(raw_data + 1)

    total_len = len(log_data)
    train_end = int(total_len * train_ratio)
    train_data = log_data[:train_end]  # (train_timesteps, nodes)

    n_timesteps, n_nodes = log_data.shape
    node_log_means = np.zeros(n_nodes)
    node_log_stds = np.zeros(n_nodes)

    for i in range(n_nodes):
        node_train_data = train_data[:, i] 
        node_mean = np.mean(node_train_data)
        node_std = np.std(node_train_data)

        if node_std < 1e-8:
            node_std = 1.0  

        node_log_means[i] = node_mean
        node_log_stds[i] = node_std

    normalized_data = np.zeros_like(log_data)
    for i in range(n_nodes):
        normalized_data[:, i] = (log_data[:, i] - node_log_means[i]) / node_log_stds[i]

    return node_log_means, node_log_stds, normalized_data

def metric_func(pred, y, times):
    result = {}

    result['MSE'], result['RMSE'], result['MAE'], result['MAPE'] = np.zeros(times), np.zeros(times), np.zeros(times), np.zeros(times)

    def cal_MAPE(pred, y):
        diff = np.abs(np.array(y) - np.array(pred))
        return np.mean(diff / y)

    for i in range(times):
        y_i = y[:,i,:]
        pred_i = pred[:,i,:]

        MSE = mean_squared_error(pred_i, y_i)
        RMSE = mean_squared_error(pred_i, y_i) ** 0.5
        MAE = mean_absolute_error(pred_i, y_i)
        result['MSE'][i] += MSE
        result['RMSE'][i] += RMSE
        result['MAE'][i] += MAE
    return result

def outputExcel(total_RMSE, total_MAE, total_MAPE):
    wb = xlwt.Workbook()
    ws = wb.add_sheet('OUTPUT')

    pred_steps = len(total_MAE)
    for i in range(pred_steps):
        ws.write(0, i, total_MAE[i])
        ws.write(1, i, total_MAPE[i]*100)
        ws.write(2, i, total_RMSE[i])

    wb.save('test.xls')


def result_print(result, logger, info_name='Evaluate'):
    total_MSE, total_RMSE, total_MAE, total_MAPE = result['MSE'], result['RMSE'], result['MAE'], result['MAPE']

    # outputExcel(total_RMSE, total_MAE, total_MAPE)

    logger.info("========== {} results ==========".format(info_name))

    pred_steps = len(total_MAE)
    mae_str = "/ ".join([f"{total_MAE[i]:.3f}" for i in range(pred_steps)])
    rmse_str = "/ ".join([f"{total_RMSE[i]:.3f}" for i in range(pred_steps)])

    logger.info(f" MAE: {mae_str}")
    # logger.info("MAPE: %.3f/ %.3f/ %.3f/ %.3f/ %.3f/ %.3f"%(total_MAPE[0] * 100, total_MAPE[1] * 100, total_MAPE[2] * 100, total_MAPE[3] * 100, total_MAPE[4] * 100, total_MAPE[5] * 100))
    logger.info(f"RMSE: {rmse_str}")
    logger.info("---------------------------------------")

    if info_name == 'Best':
        logger.info("========== Best results ==========")
        logger.info(f" MAE: {mae_str}")
        # logger.info("MAPE: %.3f/ %.3f/ %.3f/ %.3f/ %.3f/ %.3f"%(total_MAPE[0] * 100, total_MAPE[1] * 100, total_MAPE[2] * 100, total_MAPE[3] * 100, total_MAPE[4] * 100, total_MAPE[5] * 100))
        logger.info(f"RMSE: {rmse_str}")
        logger.info("---------------------------------------")

    return np.sum(total_MAE) + np.sum(total_RMSE)  


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
        X = X[:, :, :int(X.shape[2]*0.7)]
    elif stage == 'validation':
        X = X[:, :, int(X.shape[2]*0.7):int(X.shape[2]*0.8)]
    elif stage == 'test':
        X = X[:, :, int(X.shape[2]*0.8):]
    elif stage == 'source':
        X = X
    elif stage == 'target_1day':
        X = X[:, :, :288]
    elif stage == 'target_3day':
        X = X[:, :, :288*3]
    elif stage == 'target_1week':
        X = X[:, :, :288*7]
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
    D[D <= 10e-5] = 10e-5    # Prevent infs
    diag = np.reciprocal(np.sqrt(D))
    A_wave = np.multiply(np.multiply(diag.reshape((-1, 1)), A),
                         diag.reshape((1, -1)))
    return A_wave


def generate_dataset(X, num_timesteps_input, num_timesteps_output):
    # Generate the beginning index and the ending index of a sample, which
    # contains (num_points_for_training + num_points_for_predicting) points
    indices = [(i, i + (num_timesteps_input + num_timesteps_output)) for i
               in range(X.shape[2] - (
                num_timesteps_input + num_timesteps_output) + 1)] 

    # Save samples
    features, target = [], []
    for i, j in indices:
        features.append(
            X[:, :, i: i + num_timesteps_input].transpose(
                (0, 2, 1)))
        target.append(X[:, 0, i + num_timesteps_input: j])

    return torch.from_numpy(np.array(features)), \
           torch.from_numpy(np.array(target))


def setup_logger(mode, DATA, index, model):
    # Create a custom logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    # Create handlers
    c_handler = logging.StreamHandler(sys.stdout)
    f_handler = logging.FileHandler("Logs/{}_logs_{}_{}.log".format(mode, DATA, model))

    # Add handlers to the logger
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)

    return logger,"Logs/{}_logs_{}_{}.log".format(mode, DATA, model)


def rescale(data, mean ,std):
    return data*std + mean


