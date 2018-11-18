import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from utils.utils import get_immediate_files
from plot_functions import *

# log_dir = './results'
log_dir = './remote_logs'
data_set = 'wine-quality-red'

# results_dir = f'{log_dir}/{data_set}'
results_dir =  './remote_logs/bostonHousing/2018-11-18 10:42:12'

files = get_immediate_files(results_dir)
files = [f for f in files if f.split('.')[-1]=='pkl']

results = {}
split = []

for file in files:
    r = pickle.load(open(f'{results_dir}/{file}', 'rb'))
    results.update(r)
    split.append(r)

keys = ['costs', 'test_ll', 'rmses', 'noise_sigma', 'train_kl', 'train_ll']
# keys = ['costs', 'accuracies']
for key in keys:
    plot_training_curves(split, val=key)

plot_min_vs_first(split)
plot_last_vs_first(split)

# plot_max_vs_first(split, val='accuracies')
# plot_last_vs_first(split, val='accuracies')

plot_min_vs_first(split, val='rmses')
plot_last_vs_first(split, val='rmses')

plt.show()