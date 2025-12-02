def set_random_seed(seed=42):
    import random
    import numpy as np
    import torch
    import os

    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

    print(f"Random seed: {seed}")


import logging
import os
import sys
import zipfile

import numpy as np
import torch
import xlrd
import xlwt
from sklearn.metrics import mean_absolute_error, mean_squared_error


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def metric_func(pred, y, times):
    result = {}

    result['MSE'], result['RMSE'], result['MAE'], result['MAPE'] = np.zeros(times), np.zeros(times), np.zeros(
        times), np.zeros(times)

    # print("metric | pred shape:", pred.shape, " y shape:", y.shape)

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

    for i in range(len(total_MAE)):
        ws.write(0, i, total_MAE[i])
        ws.write(1, i, total_MAPE[i] * 100)
        ws.write(2, i, total_RMSE[i])
    wb.save('test.xls')


def result_print(result, logger, info_name='Evaluate'):
    total_MSE, total_RMSE, total_MAE, total_MAPE = result['MSE'], result['RMSE'], result['MAE'], result['MAPE']

    outputExcel(total_RMSE, total_MAE, total_MAPE)

    logger.info("========== {} results ==========".format(info_name))

    mae_str = "/ ".join(["%.3f"] * len(total_MAE))
    rmse_str = "/ ".join(["%.3f"] * len(total_RMSE))
    logger.info(" MAE: " + mae_str % tuple(total_MAE))
    # logger.info("MAPE: " + "/ ".join(["%.3f"] * len(total_MAPE)) % tuple(total_MAPE * 100))
    logger.info("RMSE: " + rmse_str % tuple(total_RMSE))
    logger.info("---------------------------------------")

    if info_name == 'Best':
        logger.info("========== Best results ==========")
        logger.info(" MAE: " + mae_str % tuple(total_MAE))
        # logger.info("MAPE: " + "/ ".join(["%.3f"] * len(total_MAPE)) % tuple(total_MAPE * 100))
        logger.info("RMSE: " + rmse_str % tuple(total_RMSE))
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


def generate_dataset(X, num_timesteps_input,
                     num_timesteps_output, means, stds):
    indices = [(i, i + (num_timesteps_input + num_timesteps_output))
               for i in range(X.shape[2] - (num_timesteps_input + num_timesteps_output) + 1)]
    features, target = [], []
    for i, j in indices:
        features.append(
            X[:, :, i: i + num_timesteps_input].transpose(
                (0, 2, 1)))
        target.append(X[:, 0, i + num_timesteps_input: j])

    return torch.from_numpy(np.array(features)), \
        torch.from_numpy(np.array(target))


def setup_logger(mode, DATA, index, model, aftername):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    # Create handlers
    c_handler = logging.StreamHandler(sys.stdout)

    log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{mode}_logs_{DATA}_{model}_{aftername}.log")
    f_handler = logging.FileHandler(log_path)

    # Add handlers to the logger
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)

    return logger, log_path


def rescale(data, mean, std):
    return data * std + mean

def get_device():

    # if torch.backends.mps.is_available():
    #     return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


