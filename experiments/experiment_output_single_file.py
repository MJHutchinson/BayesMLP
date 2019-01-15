import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from utils.file_utils import get_immediate_files
from plot_functions import *

log_dir = './results'
data_set = 'wine-quality-red'
# data_set = 'kin8nm'
# file = '2_layers_prior_1'
file = '2_layers_variable'
# file = '2_layers_[100, 100]_prior_sizes'
# file = '2_layers_[4, 4]_prior_sizes'

results_file = f'{log_dir}/{data_set}/{file}.pkl'

results = {}
split = []


r = pickle.load(open(results_file, 'rb'))
results.update(r)
split.append(r)

keys = ['costs', 'loglosses', 'rmses', 'noise_sigma', 'train_kl', 'train_ll']

for key in keys:
    plot_training_curves(split, val=key)

plot_min_vs_first(split)

plt.show()