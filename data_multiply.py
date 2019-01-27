import os
import datetime
import pickle
import yaml
import shutil
import argparse
import itertools

import tensorflow as tf
import data.data_loader as data
from model.regression import BayesMLPRegression
from model.test_model import test_model_regression


'''
Experiment to run a series of different parameter settings on the specified data set using the configuration specified. 
'''

parser = argparse.ArgumentParser(description='Script for dispatching train runs of BNNs over larger search spaces')
parser.add_argument('-c',  '--config', default='./config/kin8nm.yaml')
parser.add_argument('-ds', '--dataset', default='kin8nm')
parser.add_argument('-ld', '--logdir', default='./results')
parser.add_argument('-dd', '--datadir', default='./data_dir')
parser.add_argument('-cm', '--commonname', default=None)
parser.add_argument('-nd', '--nodatetime', action='store_true')
args = parser.parse_args()

experiment_config = yaml.load(open(args.config, 'rb'))

# Set up logging directory and grab the config file
date_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

if args.commonname is not None :
    if args.nodatetime:
        folder_name = args.commonname
    else:
        folder_name = f'{args.commonname}-{date_time}'
else:
    if not args.nodatetime:
        folder_name = f'{date_time}'
    else:
        raise ValueError('Must supply a common name, or set ude datetime to True')

results_dir = f'{args.logdir}/{args.dataset}/{folder_name}'

latest_dir = f'{args.logdir}/{args.dataset}/latest'

if not os.path.exists(results_dir):
    os.makedirs(results_dir)

if os.path.islink(latest_dir):
    os.unlink(latest_dir)

os.symlink(os.path.abspath(results_dir), latest_dir)

# Copy config across for reference
shutil.copy2(args.config, results_dir)

# Parameter space to sweep
hidden_sizes = experiment_config['hidden_sizes']
hidden_layers = experiment_config['hidden_layers']
learning_rates = experiment_config['learning_rates']
prior_vars = experiment_config['prior_vars']

# Additional variable
data_multiples = experiment_config['data_multiples']

# Training parameters
batch_size = experiment_config['batch_size']
epochs = experiment_config['epochs']

print(f'Running experiment on {args.dataset} with parameters:\n'
      f'{experiment_config}\n'
      f'Saving results in {results_dir}\n')


# Load in dataset and related info

data_loader = data.RegressionDataloader(args.dataset, args.datadir)
input_size, train_length, output_size = data_loader.get_dims()
_, _, y_mu, y_sigma = data_loader.get_transforms()

# Design search space for paramters
param_space = list(itertools.product(hidden_layers, hidden_sizes, learning_rates, prior_vars, data_multiples))

# Loop over parameter space
for idx, (hidden_layer, hidden_size, lr, prior_var, data_multiple) in enumerate(param_space):

    hidden_configuration = [hidden_size] * hidden_layer

    model = BayesMLPRegression(input_size, hidden_configuration, output_size, train_length*data_multiple, y_mu, y_sigma, prior_var=prior_var)

    print(f'{args.dataset} - running {model}. Parameter set {idx+1} of {len(param_space)}')

    name = f'data_multiply_{data_multiple}_{model}'
    log_dir = f'{results_dir}/logs/{name}'

    result = test_model_regression(model, data_loader, epochs, batch_size, log_freq=100, log_dir=log_dir, verbose=False)
    model.close_session()
    tf.reset_default_graph()

    model_config = model.get_config()
    train_config = {'batch_size': batch_size, 'epochs': epochs, 'results': result}
    output = {**model_config, **train_config, 'results': result}

    result_file = f'{results_dir}/{name}.pkl'
    with open(result_file, 'wb') as h:
        pickle.dump(output, h)
