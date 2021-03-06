import os
import datetime
import pickle
import yaml
import shutil
import tensorflow as tf
import argparse
from model.classification import BayesMLPClassification
from model.test_model import test_model_classification, get_activation
from utils.file_utils import num_to_name, gen_hidden_combinations, parameter_combinations
import data.data_loader as data
# Script parameters

data_set = 'mnist'
log_dir = './results'
config_dir = './config'
common_name = None


# Set up loggin directory and grab the config file
date_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

results_dir = f'{log_dir}/{data_set}/{date_time}'
latest_dir = f'{log_dir}/{data_set}/latest'
config_file = f'{config_dir}/{data_set}.yaml'


if not os.path.exists(results_dir):
    os.makedirs(results_dir)

if os.path.islink(latest_dir):
    os.unlink(latest_dir)

os.symlink(os.path.abspath(results_dir), latest_dir)

config = yaml.load(open(config_file, 'rb'))
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
data_loader = data.MnistDataloader()
input_size, train_length, output_size = data_loader.get_dims()


# Design search space for paramters
search_space = gen_hidden_combinations(search_space, hs, hidden_layers)
param_space = parameter_combinations(search_space, lrs, prior_vars)

# Loop over parameter space
for idx, (network, lr, prior_var) in enumerate(param_space):

    for activation in config['activations']:

        h = [i for i in network] # Tuple to list
        batch_size = data_loader.get_batch_size(max(h))

        logs_dir = f'{results_dir}/logs/hidden_{h}_lr_{lr}_prior_var_{prior_var}_activation_{activation}'

        print(f'running model {(network, lr, prior_var, activation)}, parameter set {idx+1} of {len(param_space)}')

        # Create model with designated parameters
        model = BayesMLPClassification(input_size, h, output_size, train_length, no_pred_samples=100, learning_rate=lr, prior_var=prior_var, activation=get_activation(activation))

        # Run a standard test on the model, logging training info etc
        result = test_model_classification(model, data_loader, epochs, batch_size, log_freq=1, log_dir=logs_dir)

        # Close model session! Important - releases VRAM, otherwise memory errors
        model.close_session()
        tf.reset_default_graph()

        # Dump results to a sensibly named file
        result = {'hidden_sizes': h, 'learning_rate': lr, 'prior_var': prior_var, 'batch_size': batch_size, 'epochs': epochs, 'results': result}
        result_file = f'{results_dir}/hidden_{h}_lr_{lr}_prior_var_{prior_var}_epochs_{epochs}_acitvation_{activation}.pkl'

        with open(result_file, 'wb') as h:
            pickle.dump(result, h)
