import os
import pickle
from tqdm import tqdm

from utils.file_utils import get_immediate_files, get_immediate_subdirectories
from utils.plot_utils import *
from collections import defaultdict

base_thesis_dir = '/home/mjhutchinson/Documents/University/4th Year/4th Year Project/Final Thesis/Thesis-LaTeX/Chapter5/Figs/'

fig_x = 3
fig_y = 2
fig_dpi = 400

do_all = False

metric_keys = ['elbo', 'test_ll', 'test_rmse', 'noise_sigma', 'train_kl', 'train_ll']

bostonHousing_results_dir = '/scratch/mjh252/logs/clean/bostonHousing/'
concrete_results_dir = '/scratch/mjh252/logs/clean/concrete/'
energy_results_dir = '/scratch/mjh252/logs/clean/energy/'
kin8nm_results_dir = '/scratch/mjh252/logs/clean/kin8nm/'
naval_results_dir = '/scratch/mjh252/logs/clean/naval-propulsion-plant/'
power_results_dir = '/scratch/mjh252/logs/clean/power-plant/'
protein_dir = '/scratch/mjh252/logs/clean/protein-tertiary-structure/'
wine_results_dir = '/scratch/mjh252/logs/clean/wine-quality-red/'
yacht_results_dir = '/scratch/mjh252/logs/clean/yacht/'

base_dump_dir = '/scratch/mjh252/summary_data/technical_milestone/'
base_load_dir = '../summary_data/'

# os.makedirs(base_dump_dir, exist_ok=True)

data = {
    'bostonHousing':   {'dim':13,  'data_size':430},
    'concrete': {'dim':8,   'data_size':875},
    'kin8nm':   {'dim':8,   'data_size':652},
    'naval':    {'dim':8,   'data_size':6963},
    'power-plant':    {'dim':16,  'data_size':10143},
    'protein-tertiary-structure':  {'dim':9,   'data_size':38870},
    'wine-quality-red':     {'dim':11,  'data_size':1359},
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
        h = list(result['hidden_sizes'][1:-2])
        h = [dim] + h + [1]

        if isinstance(result[key], np.ndarray):
            key_var = repr(list(result[key])[1:-2])
        else:
            key_var = repr(result[key])

        weights = 0.
        for idx in range(len(h) - 1):
            weights += h[idx] * h[idx + 1]

        num_weights[key_var].append(weights)
        if key == 'first_layer_size':
            layer_size[key_var].append(h[2])
        elif key == 'second_layer_size':
            layer_size[key_var].append(h[1])
        else:
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

fig_x = 5.6
fig_y = 2.8

def hyp_sweep(results_dir, data_set):

    print(f'Hidden sizes {data_set}')

    # layer_size_results = get_data(results_dir + 'sweep-hidden-sizes')
    #
    # dump_dir = base_dump_dir + f'{data_set}/'
    # os.makedirs(dump_dir, exist_ok=True)
    #
    # pickle.dump(layer_size_results, open(dump_dir + 'sweep-hidden-sizes.pkl', 'wb'))

    load_dir = base_load_dir + f'{data_set}/'
    thesis_dir = base_thesis_dir + f'{data_set}/'
    layer_size_results = pickle.load(open(load_dir + 'sweep-hidden-sizes.pkl', 'rb'))

    num_weights, layer_size, prior_var, final_ll, final_rmse, final_cost = group_by(layer_size_results,
                                                                                    data[data_set],
                                                                                    key='prior_var')

    hidden_size_fig, hidden_size_axes = plt.subplots(2,1, figsize=(text_width, text_height/2.1))

    # fig, ax = plot_dict(layer_size, final_rmse, 'Layer width', 'Final RMSE', log_scale=True)
    # fig.set_size_inches(fig_x, fig_y)
    # plt.tight_layout()
    # savefig(load_dir + f'hidden-size-rmse')
    # savefig(thesis_dir + f'hidden-size-rmse', png=False, pdf=True)
    #
    # fig, ax = plot_dict(layer_size, final_ll, 'Layer width', 'Final Log Likelihood', log_scale=True)
    # fig.set_size_inches(fig_x, fig_y)
    # plt.tight_layout()
    # savefig(load_dir + f'hidden-size-logloss')
    # savefig(thesis_dir + f'hidden-size-logloss', png=False, pdf=True)

    plot_dict(layer_size, final_rmse, 'Layer width', 'Final RMSE', log_scale=True, ax=hidden_size_axes[0])
    plot_dict(layer_size, final_ll, 'Layer width', 'Final Log Likelihood', log_scale=True, ax=hidden_size_axes[1])
    hidden_size_fig.tight_layout()

    savefig(load_dir + f'hidden-size')
    savefig(thesis_dir + f'hidden-size', png=False, pdf=True)

    print(f'Prior vars {data_set}')

    # prior_var_results = get_data(results_dir + 'sweep-prior-var')
    # pickle.dump(prior_var_results, open(dump_dir + 'sweep-prior-var.pkl', 'wb'))

    prior_var_results = pickle.load(open(load_dir + 'sweep-prior-var.pkl', 'rb'))

    num_weights, layer_size, prior_var, final_ll, final_rmse, final_cost = group_by(prior_var_results,
                                                                                    data[data_set],
                                                                                    key='hidden_sizes')

    prior_var_fig, prior_var_axes = plt.subplots(2, 1, figsize=(text_width, text_height / 2.1))

    # fig, ax = plot_dict(prior_var, final_rmse, 'Prior variance', 'Final RMSE',  log_scale=True)
    # fig.set_size_inches(fig_x, fig_y)
    # plt.tight_layout()
    # # plt.ylim(0,3)
    # savefig(load_dir + f'prior-width-rmse')
    # savefig(thesis_dir + f'prior-width-rmse', png=False, pdf=True)
    #
    # fig, ax = plot_dict(prior_var, final_ll, 'Prior variance', 'Final Log Likelihood', log_scale=True)
    # fig.set_size_inches(fig_x, fig_y)
    # plt.tight_layout()
    # # plt.ylim(-2,0)
    # savefig(load_dir + f'prior-width-logloss')
    # savefig(thesis_dir + f'prior-width-logloss', png=False, pdf=True)

    plot_dict(prior_var, final_rmse, 'Prior variance', 'Final RMSE', log_scale=True, ax=prior_var_axes[0])
    plot_dict(prior_var, final_ll, 'Prior variance', 'Final Log Likelihood', log_scale=True, ax=prior_var_axes[1])

    prior_var_fig.tight_layout()

    savefig(load_dir + f'prior-width')
    savefig(thesis_dir + f'prior-width', png=False, pdf=True)


if do_all or True:
    # hyp_sweep(bostonHousing_results_dir, 'bostonHousing')
    hyp_sweep(concrete_results_dir, 'concrete')
    # hyp_sweep(energy_results_dir, 'energy')
    # hyp_sweep(kin8nm_results_dir, 'kin8nm')
    # hyp_sweep(naval_results_dir, 'naval')
    hyp_sweep(power_results_dir, 'power-plant')
    hyp_sweep(protein_dir, 'protein-tertiary-structure')
    # hyp_sweep(wine_results_dir, 'wine-quality-red')
    # hyp_sweep(yacht_results_dir, 'yacht')



## Wine - data multiply
fig_x = 5.6
fig_y = 2.8

def data_multiply(results_dir, data_set):
    # def get_data_multiply_data(dir):
    #     files = get_immediate_files(dir)
    #     files = [f for f in files if f.split('.')[-1] == 'pkl']
    #
    #     results = []
    #
    #     for file in files:
    #         r = pickle.load(open(f'{dir}/{file}', 'rb'))
    #         r['data_multiply'] = float(file.split('_')[2])
    #         results.append(r)
    #
    #     return results

    print(f'Data multiply {data_set}')
    #
    # data_multiply_results = get_data_multiply_data(results_dir + 'data-multiply')
    # data_multiply_results = sorted(data_multiply_results, key= lambda x: x['data_multiply'])
    #
    #
    # dump_dir = base_dump_dir + f'{data_set}/'
    # os.makedirs(dump_dir, exist_ok=True)
    #
    # pickle.dump(data_multiply_results, open(dump_dir + 'data-multiply.pkl', 'wb'))

    load_dir = base_load_dir + f'{data_set}/'
    thesis_dir = base_thesis_dir + f'{data_set}/'
    data_multiply_results = pickle.load(open(load_dir + 'data-multiply.pkl', 'rb'))

    legend = [r['data_multiply'] for r in data_multiply_results]

    # fig, ax = plot_training_curves(data_multiply_results, val='elbo', title='Expected lower bound')
    # fig.set_size_inches(fig_x, fig_y)
    # plt.tight_layout()
    # savefig(load_dir + f'data-multiply-elbo')
    # savefig(thesis_dir + f'data-multiply-elbo', png=False, pdf=True)
    #
    # fig, ax = plot_training_curves(data_multiply_results, val='test_rmse', title='RMSE')
    # fig.set_size_inches(fig_x, fig_y)
    # # plt.ylim(0.55, 0.8)
    # plt.tight_layout()
    # savefig(load_dir + f'data-multiply-rmse')
    # savefig(thesis_dir + f'data-multiply-rmse', png=False, pdf=True)
    #
    # fig, ax = plot_training_curves(data_multiply_results, val='train_ll', title='Train log likelihood')
    # # plt.ylim(-4, 1)
    # fig.set_size_inches(fig_x, fig_y)
    # plt.tight_layout()
    # savefig(load_dir + f'data-multiply-train_ll')
    # savefig(thesis_dir + f'data-multiply-train_ll', png=False, pdf=True)
    #
    # fig, ax = plot_training_curves(data_multiply_results, val='test_ll', title='Test log likelihood')
    # # plt.ylim(-5, 0)
    # fig.set_size_inches(fig_x, fig_y)
    # plt.tight_layout()
    # savefig(load_dir + f'data-multiply-test_ll')
    # savefig(thesis_dir + f'data-multiply-test_ll', png=False, pdf=True)
    #
    # fig, ax = plot_training_curves(data_multiply_results, val='noise_sigma', title='Homoskedastic noise')
    # fig.set_size_inches(fig_x, fig_y)
    # plt.tight_layout()
    # savefig(load_dir + f'data-multiply-noise')
    # savefig(thesis_dir + f'data-multiply-noise', png=False, pdf=True)
    #
    # fig, ax = plot_training_curves(data_multiply_results, val='train_kl', title='KL divergence')
    # fig.set_size_inches(fig_x, fig_y)
    # plt.tight_layout()
    # savefig(load_dir + f'data-multiply-train-kl')
    # savefig(thesis_dir + f'data-multiply-train-kl', png=False, pdf=True)
    #
    # fig = plt.figure()
    # fig_legend = plt.figure(figsize=(2, 1.25))
    # ax = fig.add_subplot(111)
    # lines = [range(2)] * len(legend)
    # lines = ax.plot(*lines, *lines)
    # fig_legend.legend(lines, legend, title='Data augmentation factor', loc='center', frameon=False)
    # fig_legend.set_size_inches(fig_x, fig_y)
    # plt.tight_layout()
    # savefig(load_dir + f'data-multiply-legend')
    # savefig(thesis_dir + f'data-multiply-legend', png=False, pdf=True)

    fig, axes = plt.subplots(2,1, sharex=True, figsize=(text_width, text_height/2.1))
    # plot_training_curves(data_multiply_results, val='elbo', title='Expected lower bound', xlabel=False, ax=axes[0], legend='data_multiply')
    # plot_training_curves(data_multiply_results, val='test_rmse', title='RMSE', xlabel=False, ax=axes[1])
    plot_training_curves(data_multiply_results, val='train_ll', title='Train log likelihood', xlabel=False, ax=axes[0])
    # axes[2].set_ylim(-4, 1)
    plot_training_curves(data_multiply_results, val='test_ll', title='Test log likelihood', xlabel=False, ax=axes[1])
    # axes[3].set_ylim(-5, 0)
    # plot_training_curves(data_multiply_results, val='noise_sigma', title='Homoskedastic noise', xlabel=False, ax=axes[4])
    # plot_training_curves(data_multiply_results, val='train_kl', title='KL divergence', ax=axes[5])

    # axes[0].set_ylim(top=200)
    axes[1].set_ylim(axes[0].get_ylim())
    # axes[2].set_ylim(bottom=-200)
    # axes[3].set_ylim(bottom=-200)
    # axes[4].set_ylim(top=20)

    fig.tight_layout()
    axes[0].legend(legend, title='Data augmentation factor', fontsize=7) # title='Data augmentation factor'

    savefig(load_dir + f'data-multiply-combined')
    savefig(thesis_dir + f'data-multiply-combined', png=False, pdf=True)

if do_all or False:
    data_multiply(bostonHousing_results_dir, 'bostonHousing')
    data_multiply(concrete_results_dir, 'concrete')
    data_multiply(energy_results_dir, 'energy')
    data_multiply(kin8nm_results_dir, 'kin8nm')
    # data_multiply(naval_results_dir, 'naval')
    data_multiply(power_results_dir, 'power-plant')
    data_multiply(protein_dir, 'protein-tertiary-structure')
    data_multiply(wine_results_dir, 'wine-quality-red')
    data_multiply(yacht_results_dir, 'yacht')

## Wine - sigma_init

fig_x = 5.6
fig_y = 2.8

def initial_sigma_plot(results_dir, data_set):
    # def get_data_sigma_init(dir):
    #     files = get_immediate_files(dir)
    #     files = [f for f in files if f.split('.')[-1] == 'pkl']
    #
    #     results = []
    #
    #     for file in files:
    #         r = pickle.load(open(f'{dir}/{file}', 'rb'))
    #         r['sigma_init'] = float(file.split('_')[2])
    #         results.append(r)
    #
    #     return results

    print(f'Sigma init {data_set}')

    # sigma_init_results = get_data_sigma_init(results_dir + 'initial-noise')
    # sigma_init_results = sorted(sigma_init_results, key= lambda x: x['sigma_init'])
    #
    #
    # dump_dir = base_dump_dir + f'{data_set}/'
    # os.makedirs(dump_dir, exist_ok=True)
    #
    # pickle.dump(sigma_init_results, open(dump_dir + 'initial-noise.pkl', 'wb'))

    load_dir = base_load_dir + f'{data_set}/'
    thesis_dir = base_thesis_dir + f'{data_set}/'
    sigma_init_results = pickle.load(open(load_dir + 'initial-noise.pkl', 'rb'))


    legend = [r['sigma_init'] for r in sigma_init_results]

    # fig, ax = plot_training_curves(sigma_init_results, val='elbo', title='Expected lower bound')
    # fig.set_size_inches(fig_x, fig_y)
    # plt.tight_layout()
    # savefig(load_dir + f'sigma-init-elbo')
    # savefig(thesis_dir + f'sigma-init-elbo', png=False, pdf=True)
    #
    # fig, ax = plot_training_curves(sigma_init_results, val='test_rmse', title='RMSE')
    # fig.set_size_inches(fig_x, fig_y)
    # # plt.ylim(0.6, 0.8)
    # plt.tight_layout()
    # savefig(load_dir + f'sigma-init-rmse')
    # savefig(thesis_dir + f'sigma-init-rmse', png=False, pdf=True)
    #
    # fig, ax = plot_training_curves(sigma_init_results, val='train_ll', title='Train log likelihood')
    # # plt.ylim(-4, 1)
    # fig.set_size_inches(fig_x, fig_y)
    # plt.tight_layout()
    # savefig(load_dir + f'sigma-init-train-ll')
    # savefig(thesis_dir + f'sigma-init-train-ll', png=False, pdf=True)
    #
    # fig, ax = plot_training_curves(sigma_init_results, val='test_ll', title='Test log likelihood')
    # # plt.ylim(-5, 0)
    # fig.set_size_inches(fig_x, fig_y)
    # plt.tight_layout()
    # savefig(load_dir + f'sigma-init-test-ll')
    # savefig(thesis_dir + f'sigma-init-test-ll', png=False, pdf=True)
    #
    # fig, ax = plot_training_curves(sigma_init_results, val='noise_sigma', title='Homoskedastic noise')
    # fig.set_size_inches(fig_x, fig_y)
    # plt.tight_layout()
    # savefig(load_dir + f'sigma-init-noise')
    # savefig(thesis_dir + f'sigma-init-noise', png=False, pdf=True)
    #
    # fig, ax = plot_training_curves(sigma_init_results, val='train_kl', title='KL divergence')
    # fig.set_size_inches(fig_x, fig_y)
    # plt.tight_layout()
    # savefig(load_dir + f'sigma-init-train-kl')
    # savefig(thesis_dir + f'sigma-init-train-kl', png=False, pdf=True)
    #
    # fig = plt.figure()
    # fig_legend = plt.figure(figsize=(2, 1.25))
    # ax = fig.add_subplot(111)
    # lines = [range(2)] * len(legend)
    # lines = ax.plot(*lines, *lines)
    # fig_legend.legend(lines, legend, title='Initial noise value', loc='center', frameon=False)
    # fig_legend.set_size_inches(fig_x, fig_y)
    # plt.tight_layout()
    # savefig(load_dir + f'sigma-init-legend')
    # savefig(thesis_dir + f'sigma-init-legend', png=False, pdf=True)

    num_weights, layer_size, prior_var, final_ll, final_rmse, final_cost = group_by(sigma_init_results,
                                                                                    data[data_set],
                                                                                    key='sigma_init')
    sigmas = {repr(sigma): [sigma] for sigma in legend}

    # fig, ax = plot_dict(sigmas, final_rmse, 'Initial noise value', 'Final RMSE', use_legend=None, log_scale=True)
    # fig.set_size_inches(fig_x, fig_y)
    # plt.tight_layout()
    # savefig(load_dir + f'sigma-init-rmse-final')
    # savefig(thesis_dir + f'sigma-init-rmse-final', png=False, pdf=True)
    #
    # fig, ax = plot_dict(sigmas, final_ll, 'Initial noise value', 'Final Log Likelihood', use_legend=None, log_scale=True)
    # fig.set_size_inches(fig_x, fig_y)
    # plt.tight_layout()
    # savefig(load_dir + f'sigma-init-logloss-final')
    # savefig(thesis_dir + f'sigma-init-logloss-final', png=False, pdf=True)


    fig, axes = plt.subplots(5, 1, sharex=False, figsize=(text_width, text_height))

    axes[0].set_ylim(bottom=-200)
    axes[1].set_ylim(bottom=-200)
    # axes[0].set_yscale('log')
    # axes[1].set_yscale('log')

    plot_training_curves(sigma_init_results, val='test_ll', title='Test log likelihood', xlabel='Epoch', ax=axes[0], legend='sigma_init')
    plot_training_curves(sigma_init_results, val='train_ll', title='Train log likelihood', xlabel='Epoch', ax=axes[1])
    plot_training_curves(sigma_init_results, val='noise_sigma', title='Homoskedastic noise', xlabel='Epoch', ax=axes[2])

    # axes[0].set_xticks([])
    # axes[1].set_xticks([])
    # axes[0].get_shared_x_axes().join(axes[0], axes[1])
    # axes[0].get_shared_x_axes().join(axes[0], axes[2])

    plot_dict(sigmas, final_rmse, 'Initial noise value', 'Final RMSE', use_legend=None, log_scale=True, ax=axes[3])
    plot_dict(sigmas, final_ll, 'Initial noise value', 'Final Log Likelihood', use_legend=None, log_scale=True, ax=axes[4])
    # plot_training_curves(data_multiply_results, val='train_kl', title='KL divergence', ax=axes[5])

    fig.tight_layout()
    axes[0].legend(fontsize=7, ncol=2)  # title='Data augmentation factor'

    savefig(load_dir + f'sigma-init-combined')
    savefig(thesis_dir + f'sigma-init-combined', png=False, pdf=True)


if do_all or False:
    initial_sigma_plot(bostonHousing_results_dir, 'bostonHousing')
    # initial_sigma_plot(concrete_results_dir, 'concrete')
    # initial_sigma_plot(kin8nm_results_dir, 'kin8nm')
    # initial_sigma_plot(naval_results_dir, 'naval')
    # initial_sigma_plot(power_results_dir, 'power')
    # initial_sigma_plot(protein_dir, 'protein')
    initial_sigma_plot(wine_results_dir, 'wine-quality-red')
    initial_sigma_plot(yacht_results_dir, 'yacht')



## Variable layer size

fig_x = 5.6
fig_y = 2.8

def multi_layer_plot(results_dir, data_set):

    print(f'Multilayer plot {data_set}')

    # data_variable_size = get_data(results_dir + 'variable-layer-sizes')
    # for result in data_variable_size:
    #     result['first_layer_size'] = result['hidden_size'][0]
    #     result['second_layer_size'] = result['hidden_size'][1] if len(result['hidden_size']) > 1 else 0
    #
    #
    # dump_dir = base_dump_dir + f'{data_set}/'
    # os.makedirs(dump_dir, exist_ok=True)
    #
    # pickle.dump(data_variable_size, open(dump_dir + 'variable-layer-sizes.pkl', 'wb'))

    load_dir = base_load_dir + f'{data_set}/'
    thesis_dir = base_thesis_dir + f'{data_set}/'
    data_variable_size = pickle.load(open(load_dir + 'variable-layer-sizes.pkl', 'rb'))


    num_weights, layer_size, prior_var, final_ll, final_rmse, final_cost = group_by(data_variable_size,
                                                                                    data[data_set],
                                                                                    key='first_layer_size')

    fig, ax = plot_dict(layer_size, final_rmse, 'Layer size', 'Final RMSE', log_scale=True)
    fig.set_size_inches(fig_x, fig_y)
    ax.set_xlabel('Second layer size')
    plt.tight_layout()
    savefig(load_dir + f'variable-layer-first-layer-rmse')
    savefig(thesis_dir + f'variable-layer-first-layer-rmse', png=False, pdf=True)

    fig, ax = plot_dict(layer_size, final_ll, 'Layer size', 'Final Log Likelihood', log_scale=True)
    fig.set_size_inches(fig_x, fig_y)
    ax.set_xlabel('Second layer size')
    plt.tight_layout()
    savefig(load_dir + f'variable-layer-first-layer-logloss')
    savefig(thesis_dir + f'variable-layer-first-layer-logloss', png=False, pdf=True)


    num_weights, layer_size, prior_var, final_ll, final_rmse, final_cost = group_by(data_variable_size,
                                                                                    data[data_set],
                                                                                    key='second_layer_size')

    fig, ax = plot_dict(layer_size, final_rmse, 'Layer size', 'Final RMSE', log_scale=True)
    fig.set_size_inches(fig_x, fig_y)
    ax.set_xlabel('First layer size')
    plt.tight_layout()
    savefig(load_dir + f'variable-layer-second-layer-rmse')
    savefig(thesis_dir + f'variable-layer-second-layer-rmse', png=False, pdf=True)

    fig, ax = plot_dict(layer_size, final_ll, 'Layer size', 'Final Log Likelihood', log_scale=True)
    fig.set_size_inches(fig_x, fig_y)
    ax.set_xlabel('First layer size')
    plt.tight_layout()
    savefig(load_dir + f'variable-layer-second-layer-logloss')
    savefig(thesis_dir + f'variable-layer-second-layer-logloss', png=False, pdf=True)

if do_all or False:
    multi_layer_plot(bostonHousing_results_dir, 'bostonHousing')
    # multi_layer_plot(concrete_results_dir, 'concrete')
    # multi_layer_plot(kin8nm_results_dir, 'kin8nm')
    # multi_layer_plot(naval_results_dir, 'naval')
    # multi_layer_plot(power_results_dir, 'power')
    # multi_layer_plot(protein_dir, 'protein')
    multi_layer_plot(wine_results_dir, 'wine-quality-red')
    multi_layer_plot(yacht_results_dir, 'yacht')




plt.close('all')
