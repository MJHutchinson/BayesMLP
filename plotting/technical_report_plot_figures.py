import pickle
from utils.file_utils import get_immediate_files, get_immediate_subdirectories
from utils.plot_utils import *
from collections import defaultdict

save_dir = '/home/mjhutchinson/Documents/University/4th Year Project/Technical Milestone/figures/'

fig_x = 3
fig_y = 2
fig_dpi = 400

do_all = False

metric_keys = ['elbo', 'test_ll', 'test_rmse', 'noise_sigma', 'train_kl', 'train_ll']

bostonHousing_results_dir = './remote_logs_clean/bostonHousing/'
concrete_results_dir = './remote_logs/concrete/'
kin8nm_results_dir = './remote_logs_clean/kin8nm/'
naval_results_dir = './remote_logs/naval-propulsion-plant/'
power_results_dir = './remote_logs/power-plant/'
protein_dir = './remote_logs/protein-tertiary-structure/'
wine_results_dir = './remote_logs_clean/wine-quality-red/'
yacht_results_dir = './remote_logs_clean/yacht/'

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

    for file in files:
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

## Hypers sweeps

fig_x = 4
fig_y = 3

def hyp_sweep(results_dir, data_set):
    layer_size_results = get_data(results_dir + 'sweep-hidden-sizes')
    num_weights, layer_size, prior_var, final_ll, final_rmse, final_cost = group_by(layer_size_results,
                                                                                    data[data_set],
                                                                                    key='prior_var')
    fig, ax = plot_dict(num_weights, final_rmse, 'Weights in network', 'Final RMSE', title=data_set, log_scale=True)
    fig.set_size_inches(fig_x, fig_y)
    plt.tight_layout()

    prior_var_results = get_data(results_dir + 'sweep-prior-var')
    num_weights, layer_size, prior_var, final_ll, final_rmse, final_cost = group_by(prior_var_results,
                                                                                    data[data_set],
                                                                                    key='hidden_size')
    fig, ax = plot_dict(prior_var, final_rmse, 'Prior variance', 'Final RMSE', title=data_set, log_scale=True)
    fig.set_size_inches(fig_x, fig_y)
    plt.tight_layout()

## Boston
if do_all or True:
    hyp_sweep(bostonHousing_results_dir, 'boston')

## Wine
if do_all or True:
    hyp_sweep(wine_results_dir, 'wine')

## Yacht
if do_all or True:
    hyp_sweep(yacht_results_dir, 'yacht')

## KIN8NM
if do_all or True:
    hyp_sweep(kin8nm_results_dir, 'kin8nm')



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

    wine_data_multiply_results = get_data_multiply_data(wine_results_dir + 'data-multiply')
    wine_data_multiply_results = sorted(wine_data_multiply_results, key= lambda x: x['data_multiply'])
    legend = [r['data_multiply'] for r in wine_data_multiply_results]

    fig, ax = plot_training_curves(wine_data_multiply_results, val='elbo', title='Expected lower bound')
    fig.set_size_inches(fig_x, fig_y)
    plt.tight_layout()
    fig.savefig(save_dir + 'data-multiply/elbo.eps', dpi=fig_dpi, format='eps')

    fig, ax = plot_training_curves(wine_data_multiply_results, val='test_rmse', title='RMSE')
    fig.set_size_inches(fig_x, fig_y)
    plt.tight_layout()
    fig.savefig(save_dir + 'data-multiply/rmse.eps', dpi=fig_dpi, format='eps')

    fig, ax = plot_training_curves(wine_data_multiply_results, val='train_ll', title='Train log likelihood')
    plt.ylim(-4, 1)
    fig.set_size_inches(fig_x, fig_y)
    plt.tight_layout()
    fig.savefig(save_dir + 'data-multiply/train_ll.eps', dpi=fig_dpi, format='eps')

    fig, ax = plot_training_curves(wine_data_multiply_results, val='test_ll', title='Test log likelihood')
    plt.ylim(-5, 0)
    fig.set_size_inches(fig_x, fig_y)
    plt.tight_layout()
    fig.savefig(save_dir + 'data-multiply/test_ll.eps', dpi=fig_dpi, format='eps')

    fig, ax = plot_training_curves(wine_data_multiply_results, val='noise_sigma', title='Homoskedastic noise')
    fig.set_size_inches(fig_x, fig_y)
    plt.tight_layout()
    fig.savefig(save_dir + 'data-multiply/noise.eps', dpi=fig_dpi, format='eps')

    fig, ax = plot_training_curves(wine_data_multiply_results, val='train_kl', title='KL divergence')
    fig.set_size_inches(fig_x, fig_y)
    plt.tight_layout()
    fig.savefig(save_dir + 'data-multiply/train_kl.eps', dpi=fig_dpi, format='eps')

    fig = plt.figure()
    fig_legend = plt.figure(figsize=(2, 1.25))
    ax = fig.add_subplot(111)
    lines = [range(2)] * len(legend)
    lines = ax.plot(*lines, *lines)
    fig_legend.legend(lines, legend, title='Data augmentation factor', loc='center', frameon=False)
    fig_legend.set_size_inches(fig_x, fig_y)
    plt.tight_layout()
    fig_legend.savefig(save_dir + 'data-multiply/legend.eps', dpi=fig_dpi, format='eps')




plt.show()
