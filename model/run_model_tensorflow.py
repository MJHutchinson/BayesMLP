import os
import datetime
import pickle
import yaml
import shutil
import argparse
import itertools

import tensorflow as tf
import data.data_loader as data
from model.regression import BayesMLPRegression, BayesMLPNNRegression
from model.classification import BayesMLPNNClassification
from model.test_model import test_model_regression, test_model_classification_optim_steps
from model.neural_network_representation import MultiLayerPerceptron, get_mlp_layer_labels
from scipy.sparse import dok_matrix


def compute_validation_error(point, data_loader, params, gpu_id, results_dir, name_prefix=None):

    # with tf.device(deviceStr):

    nn = point[0]
    model_parameters = point[1]

    if data_loader.type == 'regression':
        model = BayesMLPNNRegression(data_loader.input_size, nn, data_loader.output_size, data_loader.train_length,
                                     **model_parameters)
        print(f'gpu {gpu_id} - {data_loader.pickle_name} - running {model}.')
        result = test_model_regression(model, data_loader, params['epochs'], params['batchSize'], log_freq=100,
                                       results_dir=results_dir, verbose=True, name_prefix=name_prefix)

    elif data_loader.type == 'classification':
        model = BayesMLPNNClassification(data_loader.input_size, nn, data_loader.output_size, data_loader.train_length,
                                     **model_parameters)
        print(f'gpu {gpu_id} - {data_loader.pickle_name} - running {model}.')
        result = test_model_classification_optim_steps(model, data_loader, params['epochs'], params['batchSize'], log_freq=50,
                                       results_dir=results_dir, verbose=True, name_prefix=name_prefix)

    model.close_session()
    tf.reset_default_graph()

    rolling_score = result[params['metric']][-20:]

    return sum(rolling_score)/len(rolling_score) # return the log dir too


