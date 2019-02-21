import os
import yaml
import datetime
import argparse

import numpy as np
import tensorflow as tf
import data.data_loader as data

from copy import deepcopy
from argparse import Namespace
from utils.nn_utils import get_feedforward_nn
from utils.reporters import get_reporter
from model.test_model import test_model_regression
from model.regression import BayesMLPNNRegression
from utils.mutli_gpu_runner import MultiGPURunner
from opt.bnn_function_caller import BNNMLPFunctionCaller


parser = argparse.ArgumentParser(description='Script for dispatching train runs of BNNs over larger search spaces')
parser.add_argument('-ld', '--logdir')
parser.add_argument('-dd', '--datadir')
parser.add_argument('-ds', '--dataset')
parser.add_argument('-c',  '--config')
parser.add_argument('-cm', '--commonname', default=None)
parser.add_argument('-nd', '--nodatetime', action='store_true')
args = parser.parse_args()

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

model_config = yaml.load(open(args.config, 'rb'))

n_splits = int(np.loadtxt(os.path.join(args.datadir, args.dataset, 'n_splits.txt')))

nn = get_feedforward_nn(model_config['hidden_size'], model_config['hidden_layer'])

for split in range(n_splits):
    dataloader = data.RegressionDataloaderVariableSplits(args.datadir, args.dataset, split)

    model = BayesMLPNNRegression(dataloader.input_size, nn, dataloader.output_size, dataloader.train_length,
                                 prior_var=model_config['prior_var'],
                                 hyperprior=model_config['hyperprior'])

    test_model_regression(model, dataloader, model_config['epochs'], model_config['batch_size'], 100, results_dir, f'split_{split}')

    model.close_session()
    tf.reset_default_graph()