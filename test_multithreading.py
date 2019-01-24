import os
import yaml
import time
import itertools

from argparse import Namespace
from datetime import datetime

from opt.worker_manager import GPUWorkerManager
from opt.bnn_function_caller import BNNMLPFunctionCaller
from opt.nn_function_caller import NNFunctionCaller
from utils.reporters import get_reporter
from utils.nn_utils import get_feedforward_adj_mat
from model.neural_network_representation import get_mlp_layer_labels, MultiLayerPerceptron

DATASET = 'wine-quality-red' # Dataset to run on
CONFIG = f'./config/parameter_sweep_layer_sizes/{DATASET}.yaml'

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
    'epochs': 100, #epochs,
    'learningRate': 0.001
}

GPU_IDS = [0]

EXP_DIR = f'./results/{DATASET}/multithreading-test_{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'
LOG_DIR = EXP_DIR
LOG_FILE = os.path.join(EXP_DIR, 'log')
TMP_DIR = os.path.join(EXP_DIR, 'tmp')
DATA_DIR = './data_dir'
os.mkdir(EXP_DIR)
os.mkdir(TMP_DIR)

latest_dir = f'./results/{DATASET}/latest'
if os.path.islink(latest_dir):
    os.unlink(latest_dir)
os.symlink(os.path.abspath(EXP_DIR), latest_dir)

REPORTER = get_reporter(open(LOG_FILE, 'w'))

worker_manager = GPUWorkerManager(GPU_IDS, TMP_DIR, LOG_DIR)

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

    hidden_configuration = [hidden_size] * hidden_layer
    all_layer_labels = get_mlp_layer_labels('reg')
    layer_labels = ['ip'] + ['relu'] * len(hidden_configuration) + ['linear', 'op']
    num_units_each_layer = [None] + hidden_configuration + [None, None]
    A = get_feedforward_adj_mat(len(layer_labels))

    nn = MultiLayerPerceptron('reg', layer_labels, A, num_units_each_layer, all_layer_labels)

    points.append(nn)


def wait_till_free():
    keep_looping = True
    while keep_looping:
        last_receive_time = worker_manager.one_worker_is_free()
        if last_receive_time is not None:
            # Get the latest set of results and dispatch the next job.
            # self.set_curr_spent_capital(last_receive_time)
            latest_results = worker_manager.fetch_latest_results()
            for qinfo_result in latest_results:
                print(f'Finished model {qinfo_result}')
            keep_looping = False
        else:
            time.sleep(0.5)


while len(points) > 0:

    wait_till_free()
    point = points[0]
    points.remove(point)
    qinfo = Namespace(point=point, send_time=time.time())
    worker_manager.dispatch_single_job(func_caller, point, qinfo)

