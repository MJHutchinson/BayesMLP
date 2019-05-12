import os
import pickle
from tqdm import tqdm

from utils.file_utils import get_immediate_files, get_immediate_subdirectories
from utils.plot_utils import *
from collections import defaultdict

save_dir = '/scratch/mjh252/figs/Final Thesis/Chapter5/Figs'

fig_x = 3
fig_y = 2
fig_dpi = 400

do_all = False

metric_keys = ['elbo', 'test_ll', 'test_rmse', 'noise_sigma', 'train_kl', 'train_ll']

bostonHousing_results_dir = '/scratch/mjh252/logs/clean/bostonHousing/'
concrete_results_dir = '/scratch/mjh252/logs/clean/concrete/'
kin8nm_results_dir = '/scratch/mjh252/logs/clean/kin8nm/'
naval_results_dir = '/scratch/mjh252/logs/clean/naval-propulsion-plant/'
power_results_dir = '/scratch/mjh252/logs/clean/power-plant/'
protein_dir = '/scratch/mjh252/logs/clean/protein-tertiary-structure/'
wine_results_dir = '/scratch/mjh252/logs/clean/wine-quality-red/'
yacht_results_dir = '/scratch/mjh252/logs/clean/yacht/'

base_dump_dir = '/scratch/mjh252/summary_data/technical_milestone/'
os.makedirs(base_dump_dir, exist_ok=True)

data = {
    'boston':   {'dim':13,  'data_size':430},
    'concrete': {'dim':8,   'data_size':875},
    'kin8nm':   {'dim':8,   'data_size':652},
    'naval':    {'dim':8,   'data_size':6963},
    'power':    {'dim':16,  'data_size':10143},
    'protein':  {'dim':9,   'data_size':38870},
    'wine':     {'dim':11,  'data_size':1359},
    'yacht':    {'dim':6,   'data_size':261}
}


def get_data(dir):
    files = get_immediate_files(dir)
    files = [f for f in files if f.split('.')[-1] == 'pkl']

    results = []

    for file in tqdm(files, desc=dir):
        r = pickle.load(open(f'{dir}/{file}', 'rb'))
        results.append(r)

    return results

def plot_results(data_set, results):
    dim = data_set['dim']

    for key in metric_keys:
        plot_training_curves(results, val=key)

    num_weights = defaultdict(list)
    layer_size = defaultdict(list)
    final_ll = defaultdict(list)
    final_rmse = defaultdict(list)
    final_cost = defaultdict(list)

    for result in results:
        h = result['hidden_size']
        h = [dim] + h + [1]

        prior_var = result['prior_var']

        weights = 0.
        for idx in range(len(h)-1):
            weights += h[idx]*h[idx+1]

        num_weights[prior_var].append(weights)
        layer_size[prior_var].append(h[1])
        final_ll[prior_var].append(result['results']['test_ll'][-1])
        final_rmse[prior_var].append(result['results']['test_rmse'][-1])
        final_cost[prior_var].append(result['results']['elbo'][-1])

    plot_dict(num_weights, final_ll, 'num weights', 'final ll', log_scale=True)
    plot_dict(num_weights, final_rmse, 'num weights', 'final rmse', log_scale=True)
    plot_dict(num_weights, final_cost, 'num weights', 'final cost', log_scale=True)

    rank_final_value(results, value='test_ll', minimum=False)
    rank_final_value(results, value='test_rmse', minimum=True)

    plt.show()


def group_by(results, data, key):

    dim = data['dim']
    def mean(list):
        return sum(list)/len(list)

    num_weights = defaultdict(list)
    layer_size = defaultdict(list)
    prior_var = defaultdict(list)
    final_ll = defaultdict(list)
    final_rmse = defaultdict(list)
    final_cost = defaultdict(list)

    for result in results:
        h = result['hidden_size']
        h = [dim] + h + [1]

        key_var = repr(result[key])

        weights = 0.
        for idx in range(len(h) - 1):
            weights += h[idx] * h[idx + 1]

        num_weights[key_var].append(weights)
        layer_size[key_var].append(h[1])
        prior_var[key_var].append(result['prior_var'])
        final_ll[key_var].append(result['results']['test_ll'][-1])
        final_rmse[key_var].append(result['results']['test_rmse'][-1])
        final_cost[key_var].append(result['results']['elbo'][-1])

    return num_weights, layer_size, prior_var, final_ll, final_rmse, final_cost

# plot_results(data['boston'])
# plot_results(data['concrete'])
# plot_results(data['kin8nm'])
# plot_results(data['naval'])
# plot_results(data['power'])
# plot_results(data['protein'])
# plot_results(data['wine'], wine_data_multiply_results)
# plot_results(data['yacht'])

## Compare train curves

## Hypers sweeps

fig_x = 4
fig_y = 2

def hyp_sweep(results_dir, data_set):

    print(f'Hidden sizes {data_set}')

    layer_size_results = get_data(results_dir + 'sweep-hidden-sizes')

    dump_dir = base_dump_dir + f'{data_set}/'
    os.makedirs(dump_dir, exist_ok=True)

    pickle.dump(layer_size_results, open(dump_dir + 'sweep-hidden-sizes.pkl', 'wb'))

    # num_weights, layer_size, prior_var, final_ll, final_rmse, final_cost = group_by(layer_size_results,
    #                                                                                 data[data_set],
    #                                                                                 key='prior_var')
    #
    # fig, ax = plot_dict(num_weights, final_rmse, 'Weights in network', 'Final RMSE', log_scale=True)
    # fig.set_size_inches(fig_x, fig_y)
    # plt.tight_layout()
    # fig.savefig(save_dir + f'hyp-sweep/hidden-{data_set}-rmse.eps', dpi=fig_dpi, format='eps')
    #
    # fig, ax = plot_dict(num_weights, final_ll, 'Weights in network', 'Final Log Likelihood', log_scale=True)
    # fig.set_size_inches(fig_x, fig_y)
    # plt.tight_layout()
    # fig.savefig(save_dir + f'hyp-sweep/hidden-{data_set}-logloss.eps', dpi=fig_dpi, format='eps')
    #
    #

    print(f'Prior vars {data_set}')

    prior_var_results = get_data(results_dir + 'sweep-prior-var')
    pickle.dump(prior_var_results, open(dump_dir + 'sweep-prior-var.pkl', 'wb'))

    # num_weights, layer_size, prior_var, final_ll, final_rmse, final_cost = group_by(prior_var_results,
    #                                                                                 data[data_set],
    #                                                                                 key='hidden_size')
    #
    # fig, ax = plot_dict(prior_var, final_rmse, 'Prior variance', 'Final RMSE',  log_scale=True)
    # fig.set_size_inches(fig_x, fig_y)
    # plt.tight_layout()
    # plt.ylim(0,3)
    # fig.savefig(save_dir + f'hyp-sweep/prior-{data_set}-rmse.eps', dpi=fig_dpi, format='eps')
    #
    # fig, ax = plot_dict(prior_var, final_ll, 'Prior variance', 'Final Log Likelihood', log_scale=True)
    # fig.set_size_inches(fig_x, fig_y)
    # plt.tight_layout()
    # plt.ylim(-2,0)
    # fig.savefig(save_dir + f'hyp-sweep/prior-{data_set}-logloss.eps', dpi=fig_dpi, format='eps')


if do_all or True:
    hyp_sweep(bostonHousing_results_dir, 'boston')
    hyp_sweep(concrete_results_dir, 'concrete')
    hyp_sweep(kin8nm_results_dir, 'kin8nm')
    hyp_sweep(naval_results_dir, 'naval')
    hyp_sweep(power_results_dir, 'power')
    hyp_sweep(protein_dir, 'protein')
    hyp_sweep(wine_results_dir, 'wine')
    hyp_sweep(yacht_results_dir, 'yacht')



## Wine - data multiply

fig_x = 3
fig_y = 2

if do_all or False:
    def get_data_multiply_data(dir):
        files = get_immediate_files(dir)
        files = [f for f in files if f.split('.')[-1] == 'pkl']

        results = []

        for file in files:
            r = pickle.load(open(f'{dir}/{file}', 'rb'))
            r['data_multiply'] = float(file.split('_')[2])
            results.append(r)

        return results

    results_dir = wine_results_dir
    data_set = 'wine'

    print(f'Data multiply {data_set}')

    data_multiply_results = get_data_multiply_data(results_dir + 'data-multiply')
    data_multiply_results = sorted(data_multiply_results, key= lambda x: x['data_multiply'])


    dump_dir = base_dump_dir + f'{data_set}/'
    os.makedirs(dump_dir, exist_ok=True)

    pickle.dump(data_multiply_results, open(dump_dir + 'data-multiply.pkl', 'wb'))


    # legend = [r['data_multiply'] for r in data_multiply_results]
    #
    # fig, ax = plot_training_curves(data_multiply_results, val='elbo', title='Expected lower bound')
    # fig.set_size_inches(fig_x, fig_y)
    # plt.tight_layout()
    # fig.savefig(save_dir + 'data-multiply/elbo.eps', dpi=fig_dpi, format='eps')
    #
    # fig, ax = plot_training_curves(data_multiply_results, val='test_rmse', title='RMSE')
    # fig.set_size_inches(fig_x, fig_y)
    # # plt.ylim(0.55, 0.8)
    # plt.tight_layout()
    # fig.savefig(save_dir + 'data-multiply/rmse.eps', dpi=fig_dpi, format='eps')
    #
    # fig, ax = plot_training_curves(data_multiply_results, val='train_ll', title='Train log likelihood')
    # plt.ylim(-4, 1)
    # fig.set_size_inches(fig_x, fig_y)
    # plt.tight_layout()
    # fig.savefig(save_dir + 'data-multiply/train_ll.eps', dpi=fig_dpi, format='eps')
    #
    # fig, ax = plot_training_curves(data_multiply_results, val='test_ll', title='Test log likelihood')
    # plt.ylim(-5, 0)
    # fig.set_size_inches(fig_x, fig_y)
    # plt.tight_layout()
    # fig.savefig(save_dir + 'data-multiply/test_ll.eps', dpi=fig_dpi, format='eps')
    #
    # fig, ax = plot_training_curves(data_multiply_results, val='noise_sigma', title='Homoskedastic noise')
    # fig.set_size_inches(fig_x, fig_y)
    # plt.tight_layout()
    # fig.savefig(save_dir + 'data-multiply/noise.eps', dpi=fig_dpi, format='eps')
    #
    # fig, ax = plot_training_curves(data_multiply_results, val='train_kl', title='KL divergence')
    # fig.set_size_inches(fig_x, fig_y)
    # plt.tight_layout()
    # fig.savefig(save_dir + 'data-multiply/train_kl.eps', dpi=fig_dpi, format='eps')
    #
    # fig = plt.figure()
    # fig_legend = plt.figure(figsize=(2, 1.25))
    # ax = fig.add_subplot(111)
    # lines = [range(2)] * len(legend)
    # lines = ax.plot(*lines, *lines)
    # fig_legend.legend(lines, legend, title='Data augmentation factor', loc='center', frameon=False)
    # fig_legend.set_size_inches(fig_x, fig_y)
    # plt.tight_layout()
    # fig_legend.savefig(save_dir + 'data-multiply/legend.eps', dpi=fig_dpi, format='eps')


## Wine - sigma_init

fig_x = 3
fig_y = 2

def initial_sigma_plot(results_dir, data_set):
    def get_data_sigma_init(dir):
        files = get_immediate_files(dir)
        files = [f for f in files if f.split('.')[-1] == 'pkl']

        results = []

        for file in files:
            r = pickle.load(open(f'{dir}/{file}', 'rb'))
            r['sigma_init'] = float(file.split('_')[2])
            results.append(r)

        return results

    print(f'Sigma init {data_set}')

    sigma_init_results = get_data_sigma_init(results_dir + 'initial-noise')
    sigma_init_results = sorted(sigma_init_results, key= lambda x: x['sigma_init'])


    dump_dir = base_dump_dir + f'{data_set}/'
    os.makedirs(dump_dir, exist_ok=True)

    pickle.dump(sigma_init_results, open(dump_dir + 'initial-noise.pkl', 'wb'))


    # legend = [r['sigma_init'] for r in sigma_init_results]
    #
    # fig, ax = plot_training_curves(sigma_init_results, val='elbo', title='Expected lower bound')
    # fig.set_size_inches(fig_x, fig_y)
    # plt.tight_layout()
    # fig.savefig(save_dir + f'sigma-init/{data_set}-elbo.eps', dpi=fig_dpi, format='eps')
    #
    # fig, ax = plot_training_curves(sigma_init_results, val='test_rmse', title='RMSE')
    # fig.set_size_inches(fig_x, fig_y)
    # plt.ylim(0.6, 0.8)
    # plt.tight_layout()
    # fig.savefig(save_dir + f'sigma-init/{data_set}-rmse.eps', dpi=fig_dpi, format='eps')
    #
    # fig, ax = plot_training_curves(sigma_init_results, val='train_ll', title='Train log likelihood')
    # plt.ylim(-4, 1)
    # fig.set_size_inches(fig_x, fig_y)
    # plt.tight_layout()
    # fig.savefig(save_dir + f'sigma-init/{data_set}-train_ll.eps', dpi=fig_dpi, format='eps')
    #
    # fig, ax = plot_training_curves(sigma_init_results, val='test_ll', title='Test log likelihood')
    # plt.ylim(-5, 0)
    # fig.set_size_inches(fig_x, fig_y)
    # plt.tight_layout()
    # fig.savefig(save_dir + f'sigma-init/{data_set}-test_ll.eps', dpi=fig_dpi, format='eps')
    #
    # fig, ax = plot_training_curves(sigma_init_results, val='noise_sigma', title='Homoskedastic noise')
    # fig.set_size_inches(fig_x, fig_y)
    # plt.tight_layout()
    # fig.savefig(save_dir + f'sigma-init/{data_set}-noise.eps', dpi=fig_dpi, format='eps')
    #
    # fig, ax = plot_training_curves(sigma_init_results, val='train_kl', title='KL divergence')
    # fig.set_size_inches(fig_x, fig_y)
    # plt.tight_layout()
    # fig.savefig(save_dir + f'sigma-init/{data_set}-train_kl.eps', dpi=fig_dpi, format='eps')
    #
    # fig = plt.figure()
    # fig_legend = plt.figure(figsize=(2, 1.25))
    # ax = fig.add_subplot(111)
    # lines = [range(2)] * len(legend)
    # lines = ax.plot(*lines, *lines)
    # fig_legend.legend(lines, legend, title='Initial noise value', loc='center', frameon=False)
    # fig_legend.set_size_inches(fig_x, fig_y)
    # plt.tight_layout()
    # fig_legend.savefig(save_dir + f'sigma-init/{data_set}-legend.eps', dpi=fig_dpi, format='eps')
    #
    # num_weights, layer_size, prior_var, final_ll, final_rmse, final_cost = group_by(sigma_init_results,
    #                                                                                 data[data_set],
    #                                                                                 key='sigma_init')
    # sigmas = {repr(sigma): [sigma] for sigma in legend}
    #
    # fig, ax = plot_dict(sigmas, final_rmse, 'Initial noise value', 'Final RMSE', use_legend=None, log_scale=True)
    # fig.set_size_inches(fig_x, fig_y)
    # plt.tight_layout()
    # fig.savefig(save_dir + f'sigma-init/{data_set}-sigma-rmse.eps', dpi=fig_dpi, format='eps')
    #
    # fig, ax = plot_dict(sigmas, final_ll, 'Initial noise value', 'Final Log Likelihood', use_legend=None, log_scale=True)
    # fig.set_size_inches(fig_x, fig_y)
    # plt.tight_layout()
    # fig.savefig(save_dir + f'sigma-init/{data_set}-sigma-logloss.eps', dpi=fig_dpi, format='eps')


if do_all or False:
    initial_sigma_plot(bostonHousing_results_dir, 'boston')
    initial_sigma_plot(concrete_results_dir, 'concrete')
    initial_sigma_plot(kin8nm_results_dir, 'kin8nm')
    initial_sigma_plot(naval_results_dir, 'naval')
    initial_sigma_plot(power_results_dir, 'power')
    initial_sigma_plot(protein_dir, 'protein')
    initial_sigma_plot(wine_results_dir, 'wine')
    initial_sigma_plot(yacht_results_dir, 'yacht')



## Variable layer size

fig_x = 4
fig_y = 2

def multi_layer_plot(results_dir, data_set):

    print(f'Multilayer plot {data_set}')

    data_variable_size = get_data(results_dir + 'variable-layer-sizes')
    for result in data_variable_size:
        result['first_layer_size'] = result['hidden_size'][0]
        result['second_layer_size'] = result['hidden_size'][1] if len(result['hidden_size']) > 1 else 0


    dump_dir = base_dump_dir + f'{data_set}/'
    os.makedirs(dump_dir, exist_ok=True)

    pickle.dump(data_variable_size, open(dump_dir + 'variable-layer-sizes.pkl', 'wb'))


    # num_weights, layer_size, prior_var, final_ll, final_rmse, final_cost = group_by(data_variable_size,
    #                                                                                 data[data_set],
    #                                                                                 key='first_layer_size')
    #
    # fig, ax = plot_dict(num_weights, final_rmse, 'Weights in network', 'Final RMSE', log_scale=True)
    # fig.set_size_inches(fig_x, fig_y)
    # plt.tight_layout()
    # fig.savefig(save_dir + f'variable-layer-size/first-layer-{data_set}-rmse.eps', dpi=fig_dpi, format='eps')
    #
    # fig, ax = plot_dict(num_weights, final_ll, 'Weights in network', 'Final Log Likelihood', log_scale=True)
    # fig.set_size_inches(fig_x, fig_y)
    # plt.tight_layout()
    # fig.savefig(save_dir + f'variable-layer-size/first-layer-{data_set}-logloss.eps', dpi=fig_dpi, format='eps')
    #
    #
    # num_weights, layer_size, prior_var, final_ll, final_rmse, final_cost = group_by(data_variable_size,
    #                                                                                 data[data_set],
    #                                                                                 key='second_layer_size')
    #
    # fig, ax = plot_dict(num_weights, final_rmse, 'Weights in network', 'Final RMSE', log_scale=True)
    # fig.set_size_inches(fig_x, fig_y)
    # plt.tight_layout()
    # fig.savefig(save_dir + f'variable-layer-size/second-layer-{data_set}-rmse.eps', dpi=fig_dpi, format='eps')
    #
    # fig, ax = plot_dict(num_weights, final_ll, 'Weights in network', 'Final Log Likelihood', log_scale=True)
    # fig.set_size_inches(fig_x, fig_y)
    # plt.tight_layout()
    # fig.savefig(save_dir + f'variable-layer-size/second-layer-{data_set}-logloss.eps', dpi=fig_dpi, format='eps')


if do_all or False:
    multi_layer_plot(bostonHousing_results_dir, 'boston')
    multi_layer_plot(concrete_results_dir, 'concrete')
    multi_layer_plot(kin8nm_results_dir, 'kin8nm')
    multi_layer_plot(naval_results_dir, 'naval')
    multi_layer_plot(power_results_dir, 'power')
    multi_layer_plot(protein_dir, 'protein')
    multi_layer_plot(wine_results_dir, 'wine')
    multi_layer_plot(yacht_results_dir, 'yacht')




if not do_all:
    plt.show()
