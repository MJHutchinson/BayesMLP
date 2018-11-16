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
results_dir =  './remote_logs/wine-quality-red/latest'

files = get_immediate_files(results_dir)
files = [f for f in files if f.split('.')[-1]=='pkl']

results = {}
split = []

for file in files:
    r = pickle.load(open(f'{results_dir}/{file}', 'rb'))
    results.update(r)
    split.append(r)

keys = ['costs', 'test_ll', 'rmses', 'noise_sigma', 'train_kl', 'train_ll']

for key in keys:
    plot_training_curves(split, val=key)

# plot_min_vs_first(split)

plt.show()