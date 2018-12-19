import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from utils.utils import get_immediate_files
from plot_functions import *

# log_dir = './results'
log_dir = './remote_logs'
data_set = 'bostonHousing'

# results_dir = f'{log_dir}/{data_set}'
# results_dir = './protein-tertiary-structure/2018-11-16 10:15:58'
# results_dir = './results/bostonHousing/2018-11-30 13:37:07'
results_dir = './remote_logs/wine-quality-red/2018-11-16 03:39:14'
# results_dir = './remote_logs/bostonHousing/2018-11-18 10:42:12'

files = get_immediate_files(results_dir)
files = [f for f in files if f.split('.')[-1]=='pkl']

results = {}
split = []

for file in files:
    r = pickle.load(open(f'{results_dir}/{file}', 'rb'))
    if (r['prior_var'] == 1):
        results.update(r)
        split.append(r)

keys = ['costs', 'test_ll', 'rmses', 'noise_sigma', 'train_kl', 'train_ll']
# keys = ['costs', 'accuracies']
for key in keys:
    plot_training_curves(split, val=key)
#
plot_min_vs_i(split, 0)
plot_min_vs_i(split, 4)
plot_min_vs_i(split, 9)

# plot_max_vs_i(split, 0, val='accuracies')
# plot_max_vs_i(split, 4, val='accuracies')
# plot_max_vs_i(split, 9, val='accuracies')


# rank_best_value(split, value='accuracies')


# plot_min_vs_first(split, val='rmses')
# plot_last_vs_first(split, val='rmses')

plt.show()