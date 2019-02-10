import os
import shutil
from utils.file_utils import get_immediate_subdirectories

os.makedirs('./figures', exist_ok=True)

dirs = get_immediate_subdirectories(os.curdir)

experiments = [
    'weight_pruning_hyperprior',
    'weight_pruning_prior_1',
]

plots = [
    'pruning_kl_1_layers.png',
    'pruning_kl_2_layers.png',
    'pruning_kl_3_layers.png',
    'test_rmse_true.png',
    'test_ll_true.png',
    'test_rmse_true-test_ll_true.png'
]

for dir in dirs:
    for exp in experiments:
        for plot in plots:
            try:
                shutil.copy(os.path.join(os.curdir, dir, exp, 'figs', plot), os.path.join(os.curdir, 'figures', f'{dir}_{exp}_{plot}'))
            except FileNotFoundError:
                pass