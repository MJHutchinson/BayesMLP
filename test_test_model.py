import os
import yaml
import time
import itertools

from argparse import Namespace
from datetime import datetime

from opt.worker_manager import GPUWorkerManager
from opt.bnn_function_caller import BNNMLPFunctionCaller
from utils.reporters import get_reporter
from utils.nn_utils import get_feedforward_nn
from utils.mutli_gpu_runner import MultiGPURunner

DATASET = 'yacht' # Dataset to run on
CONFIG = f'./config/{DATASET}.yaml'

experiment_config = yaml.load(open(CONFIG, 'rb'))

# Parameter space to sweep
hidden_sizes = experiment_config['hidden_sizes']
hidden_layers = experiment_config['hidden_layers']
# learning_rates = experiment_config['learning_rates']
# prior_vars = experiment_config['prior_vars']

# Training parameters
batch_size = experiment_config['batch_size']
epochs = experiment_config['epochs']

tf_params = {
    'batchSize': batch_size,
    'epochs': 100,#epochs,
    'learningRate': 0.001
}

# GPU_IDS = [0,1,2,3,5,6,7]
GPU_IDS = [0]

# EXP_DIR = f'/scratch/mjh252/logs/{DATASET}/multithreading-test_{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'
EXP_DIR = f'./results/{DATASET}/multithreading-test_{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'
LOG_DIR = EXP_DIR
LOG_FILE = os.path.join(EXP_DIR, 'log')
TMP_DIR = os.path.join(EXP_DIR, 'tmp')
DATA_DIR = './data_dir'
# DATA_DIR = '/scratch/mjh252/data/UCL'
os.mkdir(EXP_DIR)
os.mkdir(TMP_DIR)

# latest_dir = f'/scratch/mjh252/logs/{DATASET}/latest'
latest_dir = f'./results/{DATASET}/latest'
if os.path.islink(latest_dir):
    os.unlink(latest_dir)
os.symlink(os.path.abspath(EXP_DIR), latest_dir)

REPORTER = get_reporter(open(LOG_FILE, 'w'))
train_params = Namespace(data_set=DATASET,
                         data_dir=DATA_DIR,
                         tf_params=tf_params,
                         metric='test_rmse')
func_caller = BNNMLPFunctionCaller(DATASET, None, train_params,
                               reporter=REPORTER,
                               tmp_dir=TMP_DIR)

points = []

param_space = list(itertools.product(hidden_layers, hidden_sizes))

for (hidden_layer, hidden_size) in param_space:

    nn = get_feedforward_nn(hidden_size, hidden_layer)
    params = {
        'hyperprior': True,
        'prior_var': 1.0
    }
    points.append((nn, params))

from model.regression import BayesMLPNNRegression
from data.data_loader import RegressionDataloader
from model.test_model import test_model_regression

point = points[0]
data_loader = RegressionDataloader(DATASET, DATA_DIR)
model = BayesMLPNNRegression(data_loader.input_size, point[0], data_loader.train_length, **point[1])
test_model_regression(model, data_loader, 50000, 1000, 10, LOG_DIR)