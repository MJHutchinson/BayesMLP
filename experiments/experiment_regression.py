import os
import datetime
import pickle
import yaml
import shutil
import tensorflow as tf
import argparse
from model.regression import BayesMLPRegression, BaysMLPRegressionTFP
from model.utils import test_model_regression
from utils.utils import num_to_name, gen_hidden_combinations, parameter_combinations
from data.data_loader import RegressionDataloader

# Script parameters
data_set = 'wine-quality-red'
log_dir = '../results'
config_dir = '../config'
common_name = None

config_file = f'{config_dir}/{data_set}.yaml'
config = yaml.load(open(config_file, 'rb'))

# Set up logging directory and grab the config file
date_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

if common_name is not None:
    results_dir = f'{log_dir}/{data_set}/{common_name}-{date_time}'
else:
    results_dir = f'{log_dir}/{data_set}/{date_time}'

latest_dir = f'{log_dir}/{data_set}/latest'

if not os.path.exists(results_dir):
    os.makedirs(results_dir)

if os.path.islink(latest_dir):
    os.unlink(latest_dir)

os.symlink(os.path.abspath(results_dir), latest_dir)

# Copy config across for reference
shutil.copy2(config_file, results_dir)


# Parse configuration
hidden_layers = config['hidden_layers']
hs = config['hs']
hs = list(reversed(hs))

epochs = config['epochs']
search_space = config['search_space']
lrs = config['learning_rates']
prior_vars = config['prior_vars']


print(f'Running experiment on {data_set} with parameters:\n'
      f'{config}\n'
      f'Saving results in {results_dir}\n')


# Load in dataset and related info
data_loader = RegressionDataloader(pickle_name=data_set, data_dir='../data_dir')
input_size, train_length, output_size = data_loader.get_dims()
batch_size = config['batch_size']
_, _, y_mu, y_sigma = data_loader.get_transforms()


# Design search space for paramters
search_space = gen_hidden_combinations(search_space, hs, hidden_layers)
param_space = parameter_combinations(search_space, lrs, prior_vars)

# Loop over parameter space
for idx, (network, lr, prior_var) in enumerate(param_space):

    h = [i for i in network] # Tuple to list

    # Create model with designated parameters
    model = BayesMLPRegression(input_size, h, output_size, train_length, y_mu, y_sigma, no_train_samples=10, no_pred_samples=100, learning_rate=lr, prior_var=prior_var)

    print(f'running model {model}, parameter set {idx+1} of {len(param_space)}')

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

    with open(result_file, 'wb') as h:
        pickle.dump(result, h)
