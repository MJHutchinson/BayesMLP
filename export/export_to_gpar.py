import os
import numpy as np
from utils.results_utils import ExperimentResults

x_funcs = [
    lambda result: len(result['hidden_sizes']) - 3,
    lambda result: result['hidden_sizes'][1]
]

f_value = 'test_ll_true'
f_sample_rate = 5000

def export(results_file):
    results = ExperimentResults(results_file)

    x = None
    f = None

    for result in results.results:
        _x = np.array([f(result) for f in x_funcs])[:, np.newaxis].T
        _f = np.array([result['results'][f_value][i] for i in range(0, len(result['results'][f_value]), f_sample_rate)])[:, np.newaxis].T

        if x is None:
            x = np.array(_x)
        else:
            x = np.concatenate((x, _x), axis=0)

        if f is None:
            f = np.array(_f)
        else:
            f = np.concatenate((f, _f), axis=0)

    export_dir = os.path.join(os.curdir, *results_file.split('/')[2:])
    os.makedirs(export_dir, exist_ok=True)
    np.savetxt(os.path.join(export_dir, 'x.txt'), x, fmt='%i')
    np.savetxt(os.path.join(export_dir, 'f.txt'), f)


files = [
    '../remote_logs_clean/bostonHousing/weight_pruning_hyperprior',
    '../remote_logs_clean/concrete/weight_pruning_hyperprior',
    '../remote_logs_clean/energy/weight_pruning_hyperprior',
    '../remote_logs_clean/kin8nm/weight_pruning_hyperprior',
    # '../remote_logs_clean/naval-propulsion-plant/weight_pruning_hyperprior',
    '../remote_logs_clean/power-plant/weight_pruning_hyperprior',
    '../remote_logs_clean/protein-tertiary-structure/weight_pruning_hyperprior',
    '../remote_logs_clean/wine-quality-red/weight_pruning_hyperprior',
    '../remote_logs_clean/yacht/weight_pruning_hyperprior',
]

for file in files:
    print(file)
    export(file)