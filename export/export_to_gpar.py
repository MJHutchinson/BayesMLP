import os
import numpy as np
from utils.results_utils import ExperimentResults

x_funcs = [
    lambda result: len(result['hidden_sizes']) - 3,
    lambda result: result['hidden_sizes'][1]
]

f_sample_rate = 10000


def export(results_file):
    results = ExperimentResults(results_file)

    x_test_ll = None
    f_test_ll = None

    x_test_rmse = None
    f_test_rmse = None

    for result in results.results:
        f_samples = list(range(1000, 5000 + 1000, 1000)) + list(range(10000, len(result['results']['test_ll_true'])+f_sample_rate, f_sample_rate))

        _x_test_ll = np.array([f(result) for f in x_funcs])[:, np.newaxis].T
        _f_test_ll = np.array([np.mean(result['results']['test_ll_true'][i-51:i-1]) for i in f_samples])[:, np.newaxis].T

        _x_test_rmse = np.array([f(result) for f in x_funcs])[:, np.newaxis].T
        _f_test_rmse = np.array([np.mean(result['results']['test_rmse'][i-51:i-1]) for i in f_samples])[:, np.newaxis].T

        if x_test_ll is None:
            x_test_ll = np.array(_x_test_ll)
        else:
            x_test_ll = np.concatenate((x_test_ll, _x_test_ll), axis=0)

        if f_test_ll is None:
            f_test_ll = np.array(_f_test_ll)
        else:
            f_test_ll = np.concatenate((f_test_ll, _f_test_ll), axis=0)

        if x_test_rmse is None:
            x_test_rmse = np.array(_x_test_rmse)
        else:
            x_test_rmse = np.concatenate((x_test_rmse, _x_test_rmse), axis=0)

        if f_test_rmse is None:
            f_test_rmse = np.array(_f_test_rmse)
        else:
            f_test_rmse = np.concatenate((f_test_rmse, _f_test_rmse), axis=0)

    export_dir = os.path.join(os.curdir, *results_file.split('/')[2:])
    os.makedirs(export_dir, exist_ok=True)

    np.savetxt(os.path.join(export_dir, 'x_loglik.txt'), x_test_ll, fmt='%i')
    np.savetxt(os.path.join(export_dir, 'f_loglik.txt'), f_test_ll)
    np.savetxt(os.path.join(export_dir, 'x_rmse.txt'), x_test_rmse, fmt='%i')
    np.savetxt(os.path.join(export_dir, 'f_rmse.txt'), f_test_rmse)
    np.savetxt(os.path.join(export_dir, 'f_steps.txt'), np.array(f_samples), fmt='%i')


files = [
    '../remote_logs_clean/bostonHousing/weight_pruning_hyperprior3',
    # '../remote_logs_clean/concrete/weight_pruning_hyperprior2',
    # '../remote_logs_clean/energy/weight_pruning_hyperprior2',
    # '../remote_logs_clean/kin8nm/weight_pruning_hyperprior2',
    # '../remote_logs_clean/naval-propulsion-plant/weight_pruning_hyperprior',
    # '../remote_logs_clean/power-plant/weight_pruning_hyperprior2',
    # '../remote_logs_clean/protein-tertiary-structure/weight_pruning_hyperprior2',
    '../remote_logs_clean/wine-quality-red/weight_pruning_hyperprior3',
    '../remote_logs_clean/yacht/weight_pruning_hyperprior3',
]

for file in files:
    print(file)
    try:
        export(file)
    except Exception:
        pass