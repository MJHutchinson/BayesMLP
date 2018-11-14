import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from utils.utils import get_immediate_files
from plot_functions import *

log_dir = './results'
data_set = 'wine-quality-red'

results_dir = f'{log_dir}/{data_set}'

files = get_immediate_files(results_dir)
files = [f for f in files if f.split('.')[1]=='pkl']

results = {}
split = []

for file in files:
    r = pickle.load(open(f'{results_dir}/{file}', 'rb'))
    results.update(r)
    split.append(r)

keys = ['costs', 'loglosses', 'rmses', 'noise_sigma', 'train_kl', 'train_ll']

for key in keys:
    plot_training_curves(split, val=key)

plot_min_vs_first(split)

plt.show()