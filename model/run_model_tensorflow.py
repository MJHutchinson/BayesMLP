import os
import datetime
import pickle
import yaml
import shutil
import argparse
import itertools

import tensorflow as tf
import data.data_loader as data
from model.regression import BayesMLPRegressionHyperprior, BayesMLPRegression, BayesMLPNNRegression, BayesSkipMLPRegression, BayesMLPNNRegressionHyperprior
from model.test_model import test_model_regression
from model.neural_network_representation import MultiLayerPerceptron, get_mlp_layer_labels
from scipy.sparse import dok_matrix


def compute_validation_error(nn, data_loader, params, gpu_id, results_dir):

    # with tf.device(deviceStr):

    model = BayesMLPNNRegressionHyperprior(data_loader.input_size, nn, data_loader.train_length, data_loader.y_mu, data_loader.y_sigma, hyperprior=True)
    print(f'gpu {gpu_id} - {data_loader.pickle_name} - running {model}.')
    result = test_model_regression(model, data_loader, params['epochs'], params['batchSize'], log_freq=100, results_dir=results_dir, verbose=False)
    model.close_session()
    tf.reset_default_graph()

    rolling_score = result[params['metric']][-20:]

    return sum(rolling_score)/len(rolling_score) # return the log dir too


