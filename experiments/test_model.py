import os
import datetime
import pickle
import yaml
import shutil
import argparse
import itertools

import tensorflow as tf
import data.data_loader as data
from model.regression import BayesMLPRegressionHyperprior, BayesMLPRegression, BayesMLPNNRegression, BayesSkipMLPRegression, BayesMLPNNRegression
from model.test_model import test_model_regression
from model.neural_network_representation import MultiLayerPerceptron, get_mlp_layer_labels
from scipy.sparse import dok_matrix

'''Test a model to check perfomance'''

# Set up logging directory and grab the config file
# logdir = './results'
# datadir = './data_dir'

logdir = '/scratch/mjh252/logs/tests/'
datadir = '/scratch/mjh252/data/UCL/'

dataset = 'wine-quality-red'
date_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

commonname = 'model_test'

folder_name = f'{commonname}-{date_time}'

results_dir = f'{logdir}/{dataset}/{folder_name}'

latest_dir = f'{logdir}/{dataset}/latest'

if not os.path.exists(results_dir):
    os.makedirs(results_dir)

if os.path.islink(latest_dir):
    os.unlink(latest_dir)

os.symlink(os.path.abspath(results_dir), latest_dir)

# Parameter space to sweep
hidden_sizes = [50]
hidden_layers = [2]
learning_rates = [0.001]
prior_vars = [1.]

# Training parameters
batch_size = 1000
epochs = 100000

print(f'Running experiment on {dataset}:\n '
      f'Saving results in {results_dir}\n')


# Load in dataset and related info

data_loader = data.RegressionDataloader(dataset, datadir)
input_size, train_length, output_size = data_loader.get_dims()
_, _, y_mu, y_sigma = data_loader.get_transforms()

# Design search space for paramters
param_space = list(itertools.product(hidden_layers, hidden_sizes, learning_rates, prior_vars))

# Loop over parameter space
for idx, (hidden_layer, hidden_size, lr, prior_var) in enumerate(param_space):

    hidden_configuration = [hidden_size] * hidden_layer

    for j in range(20):
        # model = BayesMLPRegression(input_size, hidden_configuration, output_size, train_length, y_mu, y_sigma, prior_var=prior_var)

        all_layer_labels = get_mlp_layer_labels('reg')

        layer_labels = ['ip'] + ['relu'] * len(hidden_configuration) + ['linear', 'op']
        num_units_each_layer = [None] + hidden_configuration + [None, None]


        def get_feedforward_adj_mat(num_layers):
            """ Returns an adjacency matrix for a feed forward network. """
            ret = dok_matrix((num_layers, num_layers))
            for i in range(num_layers - 1):
                ret[i, i + 1] = 1
            return ret


        A = get_feedforward_adj_mat(len(layer_labels))

        nn = MultiLayerPerceptron('reg', layer_labels, A, num_units_each_layer, all_layer_labels)

        model = BayesMLPNNRegression(input_size, nn, train_length, y_mu, y_sigma, prior_var=prior_var,
                                     hyperprior=True)
        print(f'{dataset} - running {model}. Parameter set {idx + 1} of {len(param_space)}')
        test_model_regression(model, data_loader, epochs, batch_size, log_freq=100, results_dir=results_dir,
                              name_prefix=j, verbose=True)
        model.close_session()
        tf.reset_default_graph()


        model = BayesMLPNNRegression(input_size, nn, train_length, y_mu, y_sigma, prior_var=prior_var,
                                     hyperprior=False)
        print(f'{dataset} - running {model}. Parameter set {idx+1} of {len(param_space)}')
        test_model_regression(model, data_loader, epochs, batch_size, log_freq=100, results_dir=results_dir,
                              name_prefix=j, verbose=True)
        model.close_session()
        tf.reset_default_graph()



        model = BayesMLPRegressionHyperprior(input_size, hidden_configuration, output_size, train_length, y_mu, y_sigma,
                                             prior_var=prior_var, hyperprior=False)
        print(f'{dataset} - running {model}. Parameter set {idx + 1} of {len(param_space)}')
        test_model_regression(model, data_loader, epochs, batch_size, log_freq=100, results_dir=results_dir,
                              name_prefix=j, verbose=True)
        model.close_session()
        tf.reset_default_graph()



        model = BayesMLPRegressionHyperprior(input_size, hidden_configuration, output_size, train_length, y_mu, y_sigma,
                                             prior_var=prior_var, hyperprior=True)
        print(f'{dataset} - running {model}. Parameter set {idx + 1} of {len(param_space)}')
        test_model_regression(model, data_loader, epochs, batch_size, log_freq=100, results_dir=results_dir,
                              name_prefix=j, verbose=True)
        model.close_session()
        tf.reset_default_graph()
