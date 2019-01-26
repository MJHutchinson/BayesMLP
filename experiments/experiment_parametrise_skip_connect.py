import argparse
import datetime
import os
import pickle
import shutil

import tensorflow as tf
import yaml
from scipy.sparse import dok_matrix

import data.data_loader as data
from model.neural_network_representation import MultiLayerPerceptron, get_mlp_layer_labels
from model.regression import BayesMLPNNRegression
from model.test_model import test_model_regression
from utils.general_utils import _set_coords_to_val
from utils.file_utils import gen_hidden_combinations, parameter_combinations

parser = argparse.ArgumentParser(description='Script for dispatching train runs of BNNs over larger search spaces')

parser.add_argument('-c', '--config', required=True)
parser.add_argument('-ds', '--dataset', required=True)
parser.add_argument('-ld', '--logdir', default='./results')
parser.add_argument('-dd', '--datadir', default='./data_dir')
parser.add_argument('-cm', '--commonname', default=None)

args = parser.parse_args()

model_config = yaml.load(open(args.config, 'rb'))

# Script parameters
data_set = args.dataset
log_dir = args.logdir
common_name = args.commonname

# Set up logging directory and grab the config file
date_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

if common_name is not None:
    results_dir = f'{log_dir}/{data_set}/{common_name}-{date_time}'
else:
    results_dir = f'{log_dir}/{data_set}/{date_time}'

latest_dir = f'{log_dir}/{data_set}/latest'

#####


if not os.path.exists(results_dir):
    os.makedirs(results_dir)

if os.path.islink(latest_dir):
    os.unlink(latest_dir)

os.symlink(os.path.abspath(results_dir), latest_dir)

# Copy config across for reference
shutil.copy2(args.config, results_dir)

# Parse configuration
hidden_layers = model_config['hidden_layers']

hs = model_config['hs']
hs = list(reversed(hs))

epochs = model_config['epochs']
search_space = model_config['search_space']
lrs = model_config['learning_rates']
prior_vars = model_config['prior_vars']
batch_size = model_config['batch_size']
skips = model_config['skips']

print(f'Running experiment on {data_set} with parameters:\n'
      f'{model_config}\n'
      f'Saving results in {results_dir}\n')

# Load in dataset and related info
data_loader = data.RegressionDataloader(data_set, args.datadir)
input_size, train_length, output_size = data_loader.get_dims()
_, _, y_mu, y_sigma = data_loader.get_transforms()

# Design search space for paramters
search_space = gen_hidden_combinations(search_space, hs, hidden_layers)
param_space = parameter_combinations(search_space, lrs, prior_vars)


# Loop over parameter space
for idx, (network, lr, prior_var) in enumerate(param_space):

    h = [i for i in network]  # Tuple to list

    for idy, skip in enumerate(skips):

        all_layer_labels = get_mlp_layer_labels('reg')

        layer_labels = ['ip'] + ['relu'] * len(h) + ['linear', 'op']
        num_units_each_layer = [None] + h + [None, None]


        def get_feedforward_adj_mat(num_layers):
            """ Returns an adjacency matrix for a feed forward network. """
            ret = dok_matrix((num_layers, num_layers))
            for i in range(num_layers - 1):
                ret[i, i + 1] = 1
            return ret

        A = get_feedforward_adj_mat(len(layer_labels))
        _set_coords_to_val(A, skip, 1)

        nn = MultiLayerPerceptron('reg', layer_labels, A, num_units_each_layer, all_layer_labels)

        # Create model with designated parameters
        model = BayesMLPNNRegression(input_size, nn, train_length, y_mu, y_sigma, no_train_samples=10,
                                     no_pred_samples=100, learning_rate=lr, prior_var=prior_var)

        print(f'running model {model}, parameter set {idx+idy+1} of {len(param_space)*len(skip)}')

        logs_dir = f'{results_dir}/logs/{model}'

        # Run a standard test on the model, logging training info etc
        result = test_model_regression(model, data_loader, epochs, batch_size, log_freq=100, log_dir=logs_dir)

        # Close model session! Important - releases VRAM, otherwise memory errors
        model.close_session()
        tf.reset_default_graph()

        # Dump results to a sensibly named file
        model_config = model.get_config()
        train_config = {'batch_size': batch_size, 'epochs': epochs, 'results': result}
        result = {**model_config, **train_config, 'results': result}
        result_file = f'{results_dir}/{model}.pkl'

        with open(result_file, 'wb') as handle:
            pickle.dump(result, handle)
