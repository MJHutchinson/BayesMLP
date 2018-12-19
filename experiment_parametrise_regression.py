import os
import datetime
import pickle
import yaml
import shutil
import tensorflow as tf
import argparse
from model.regression import BayesMLPRegression
from model.utils import test_model_regression
from utils.utils import num_to_name, get_search_space, parameter_combinations
import data.data_loader as data

parser = argparse.ArgumentParser(description='Script for dispatching train runs of BNNs over larger search spaces')

parser.add_argument('-c', '--config', required=True)
parser.add_argument('-ds', '--dataset', required=True)
parser.add_argument('-ld', '--logdir', default='./results')
parser.add_argument('-dd', '--datadir', default='./data_dir')

args = parser.parse_args()

model_config = yaml.load(open(args.config, 'rb'))

# Script parameters
data_set = args.dataset
log_dir = args.logdir
common_name = None

# Set up loggin directory and grab the config file
date_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

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


print(f'Running experiment on {data_set} with parameters:\n'
      f'{model_config}\n'
      f'Saving results in {results_dir}\n')


# Load in dataset and related info
data_loader = data.RegressionDataloader(data_set, args.datadir)
input_size, train_length, output_size = data_loader.get_dims()
_, _, y_mu, y_sigma = data_loader.get_transforms()


# Design search space for paramters
search_space = get_search_space(search_space, hs, hidden_layers)
param_space = parameter_combinations(search_space, lrs, prior_vars)

# Loop over parameter space
for idx, (network, lr, prior_var) in enumerate(param_space):

    h = [i for i in network] # Tuple to list

    logs_dir = f'{results_dir}/logs/hidden_{h}_lr_{lr}_prior_var_{prior_var}'

    print(f'running model {(network, lr, prior_var)}, parameter set {idx+1} of {len(param_space)}')

    # Create model with designated parameters
    model = BayesMLPRegression(input_size, h, output_size, train_length, y_mu, y_sigma, no_pred_samples=100, learning_rate=lr, prior_var=prior_var)

    # Run a standard test on the model, logging training info etc
    result = test_model_regression(model, data_loader, epochs, batch_size, log_freq=100, log_dir=logs_dir)

    # Close model session! Important - releases VRAM, otherwise memory errors
    model.close_session()
    tf.reset_default_graph()

    # Dump results to a sensibly named file
    result = {'hidden_sizes': h, 'learning_rate': lr, 'prior_var': prior_var, 'batch_size': batch_size, 'epochs': epochs, 'results': result}
    result_file = f'{results_dir}/hidden_{h}_lr_{lr}_prior_var_{prior_var}_epochs_{epochs}.pkl'

    with open(result_file, 'wb') as h:
        pickle.dump(result, h)
