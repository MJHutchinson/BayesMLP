import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from utils.file_utils import get_immediate_files
from utils.plot_utils import *

# log_dir = './results'
log_dir = './remote_logs'
data_set = 'bostonHousing'

# results_dir = f'{log_dir}/{data_set}'
# results_dir = './remote_logs/bostonHousing/christsmas-sweep-1-2018-12-20 19:33:57'
# results_dir = './remote_logs/concrete/christsmas-sweep-1-2018-12-20 19:33:57'
# results_dir = './remote_logs/kin8nm/christsmas-sweep-1-2018-12-20 19:33:57'
# results_dir = './remote_logs/naval-propulsion-plant/christsmas-sweep-1-2018-12-20 19:33:57'
# results_dir = './remote_logs/power-plant/christsmas-sweep-1-2018-12-20 19:33:57'
# results_dir = './remote_logs/protein-tertiary-structure/christsmas-sweep-1-2018-12-20 19:33:57'
# results_dir = './remote_logs/wine-quality-red/christsmas-sweep-1-2018-12-20 19:33:57'
results_dir = './remote_logs_clean/bostonHousing/initial-noise'
# results_dir = './remote_logs_clean/wine-quality-red/initial-noise'
# results_dir = './remote_logs_clean/yacht/initial-noise'



# results_dir = './remote_logs/bostonHousing/skips-2018-12-30 14:59:52'

files = get_immediate_files(results_dir)
files = [f for f in files if f.split('.')[-1]=='pkl']

results = {}
split = []

for file in files:
    r = pickle.load(open(f'{results_dir}/{file}', 'rb'))
    # if (r['prior_var'] == 1):
    results.update(r)
    split.append(r)

metric_keys = ['elbo', 'test_ll', 'test_rmse', 'noise_sigma', 'train_kl', 'train_ll']
# keys = ['costs', 'accuracies']
for key in metric_keys:
    plot_training_curves(split, val=key)
#
plot_min_vs_i(split, 0, val='elbo')
plot_min_vs_i(split, 4, val='elbo')
plot_min_vs_i(split, 9, val='elbo')

# plot_max_vs_i(split, 0, val='accuracies')
# plot_max_vs_i(split, 4, val='accuracies')
# plot_max_vs_i(split, 9, val='accuracies')


rank_final_value(split, value='test_ll', minimum=False)
rank_final_value(split, value='test_rmse', minimum=True)


# plot_min_vs_first(split, val='rmses')
# plot_last_vs_first(split, val='rmses')

plt.show()