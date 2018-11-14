import pickle
from utils.utils import get_immediate_files, get_immediate_subdirectories
from plot_functions import *

log_dir = './results'
data_set = 'wine-quality-red'
experiment = None

if experiment is None:
    files = get_immediate_subdirectories(f'{log_dir}/{data_set}')
    results_dir = f'{log_dir}/{data_set}/{files[0]}'
else:
    results_dir = f'{log_dir}/{data_set}/{experiment}'


files = get_immediate_files(results_dir)
files = [f for f in files if f.split('.')[-1]=='pkl']

results = []

for file in files:
    r = pickle.load(open(f'{results_dir}/{file}', 'rb'))
    results.append(r)

keys = ['costs', 'test_ll', 'rmses', 'noise_sigma', 'train_kl', 'train_ll']

for key in keys:
    plot_training_curves(results, val=key)

# plot_min_vs_first(results)

plt.show()