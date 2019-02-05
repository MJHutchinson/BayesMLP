import os
import datetime
import pickle
import yaml
import shutil
import argparse
import itertools
from argparse import Namespace

import tensorflow as tf
import data.data_loader as data
from utils.nn_utils import get_feedforward_nn
from utils.mutli_gpu_runner import MultiGPURunner
from utils.reporters import get_reporter
from opt.bnn_function_caller import BNNMLPFunctionCaller


'''
Experiment to run a series of different parameter settings on the specified data set using the configuration specified. 
'''

parser = argparse.ArgumentParser(description='Script for dispatching train runs of BNNs over larger search spaces')
parser.add_argument('-c',  '--config')
parser.add_argument('-ds', '--dataset')
parser.add_argument('-ld', '--logdir')
parser.add_argument('-dd', '--datadir')
parser.add_argument('-cm', '--commonname', default=None)
parser.add_argument('--gpu', nargs='+', type=int)
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
hyperprior = experiment_config['hyperprior']

print(f'Running experiment on {args.dataset} with parameters:\n'
      f'{experiment_config}\n'
      f'Saving results in {results_dir}\n')


# Load in dataset and related info

data_loader = data.RegressionDataloaderFixedSplits(args.dataset, args.datadir + '/presplit')

# Design search space for paramters
param_space = list(itertools.product(hidden_layers, hidden_sizes, learning_rates, prior_vars))

points = []
# Loop over parameter space
if 'iterations' in experiment_config.keys():
    for i in range(experiment_config['iterations']):
        for idx, (hidden_layer, hidden_size, lr, prior_var) in enumerate(param_space):
            nn = get_feedforward_nn(hidden_size, hidden_layer)
            params = {
                'learning_rate': lr,
                'prior_var': prior_var,
                'hyperprior': hyperprior,
                'iteration': i
            }
            points.append((nn, params))
else:
    for idx, (hidden_layer, hidden_size, lr, prior_var) in enumerate(param_space):
        nn = get_feedforward_nn(hidden_size, hidden_layer)
        params = {
            'learning_rate': lr,
            'prior_var': prior_var,
            'hyperprior': hyperprior
        }
        points.append((nn, params))


REPORTER = get_reporter(open(os.path.join(results_dir, 'log'), 'w'))
train_params = Namespace(data_set=args.dataset,
                         data_dir=args.datadir,
                         tf_params={
                             'batchSize': experiment_config['batch_size'],
                             'epochs': experiment_config['epochs']
                         },
                         metric='test_rmse')
func_caller = BNNMLPFunctionCaller(data_loader, args.dataset, None, train_params,
                               reporter=REPORTER,
                               tmp_dir=None)

gpu_runner = MultiGPURunner(func_caller, gpu_ids=args.gpu, log_dir=results_dir, tmp_dir=os.path.join(results_dir, 'tmp'))
gpu_runner.run_points(points)
