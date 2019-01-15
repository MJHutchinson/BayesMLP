import pickle
from utils.file_utils import get_immediate_files, get_immediate_subdirectories
from plot_functions import *

save_dir = ''

metric_keys = ['costs', 'test_ll', 'rmses', 'noise_sigma', 'train_kl', 'train_ll']

bostonHousing_results_dir = './remote_logs/bostonHousing/christsmas-sweep-3-2019-01-09 08:51:54'
concrete_results_dir = './remote_logs/concrete/christsmas-sweep-3-2019-01-09 08:51:54'
kin8nm_results_dir = './remote_logs/kin8nm/christsmas-sweep-3-2019-01-09 08:51:54'
naval_results_dir = './remote_logs/naval-propulsion-plant/christsmas-sweep-3-2019-01-09 08:51:54'
power_results_dir = './remote_logs/power-plant/christsmas-sweep-3-2019-01-09 08:51:54'
protein_dir = './remote_logs/protein-tertiary-structure/christsmas-sweep-3-2019-01-09 08:51:54'
wine_results_dir = './remote_logs/wine-quality-red/christsmas-sweep-3-2019-01-09 08:51:54'
yacht_results_dir = './remote_logs/yacht/christsmas-sweep-3-2019-01-09 08:51:54'

data = {
    'boston':   {'dir':bostonHousing_results_dir, 'dim':13, 'data_size':430},
    'concrete': {'dir':concrete_results_dir, 'dim':8, 'data_size':875},
    'kin8nm':   {'dir':kin8nm_results_dir, 'dim':8, 'data_size':652},
    'naval':    {'dir':naval_results_dir, 'dim':8, 'data_size':6963},
    'power':    {'dir':power_results_dir, 'dim':16, 'data_size':10143},
    'protein':  {'dir':protein_dir, 'dim':9, 'data_size':38870},
    'wine':     {'dir':wine_results_dir, 'dim':11, 'data_size':1359},
    'yacht':    {'dir':yacht_results_dir, 'dim':6, 'data_size':261}
}

for key in data.keys():
    data_set = data[key]
    dir = data_set['dir']

    files = get_immediate_files(dir)
    files = [f for f in files if f.split('.')[-1] == 'pkl']

    results = []

    for file in files:
        r = pickle.load(open(f'{dir}/{file}', 'rb'))
        results.append(r)

    data_set['results'] = results

    data[key] = data_set


def plot_results(data_set):
    results = data_set['results']
    dim = data_set['dim']

    for key in metric_keys:
        plot_training_curves(results, val=key)

    num_weights = []
    layer_size = []
    final_ll = []
    final_rmse = []
    final_cost = []

    for result in results:
        h = result['hidden_size']
        h = [dim] + h + [1]

        weights = 0.
        for idx in range(len(h)-1):
            weights += h[idx]*h[idx+1]

        num_weights.append(weights)
        layer_size.append(h[1])
        final_ll.append(result['results']['train_ll'][-1])
        final_rmse.append(result['results']['rmses'][-1])
        final_cost.append(result['results']['costs'][-1])

    plot_xy(num_weights, final_ll, 'num weights', 'final ll')
    plot_xy(num_weights, final_rmse, 'num weights', 'final rmse')
    plot_xy(num_weights, final_cost, 'num weights', 'final cost')

    rank_final_value(results, value='test_ll', minimum=False)
    rank_final_value(results, value='rmses', minimum=True)

    plt.show()


# plot_results(data['boston'])
# plot_results(data['concrete'])
# plot_results(data['kin8nm'])
# plot_results(data['naval'])
# plot_results(data['power'])
# plot_results(data['protein'])
# plot_results(data['wine'])
plot_results(data['yacht'])
