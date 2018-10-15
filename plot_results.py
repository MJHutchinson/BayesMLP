import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from utils.utils import get_immediate_files

log_dir = './results'
data_set = 'protein-tertiary-structure'

results_dir = f'{log_dir}/{data_set}'

files = get_immediate_files(results_dir)
files = [f for f in files if f.split('.')[1]=='pkl']

results = {}
split = []
columns = ['File', 'Network', 'Epochs', 'Costs', 'Accuracies']
df = pd.DataFrame(columns=columns)

for file in files:
    r = pickle.load(open(f'{results_dir}/{file}', 'rb'))
    results.update(r)
    split.append(r)
    print(r)


def plot_training_curves(*input, legend=None):
    _, ax = plt.subplots(1, 1)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    legend = []
    for results in input:
        for key in results.keys():
            result = results[key]['results']
            ax.plot(result['accuracies'])
            legend.append(key)

    # if legend is not None:
    #     ax.legend(legend)


def plot_best_vs_first(*input, legend=None):
    _, ax = plt.subplots(1, 1)
    ax.set_xlabel('First epoch accuracy')
    ax.set_ylabel('Best accuracy')
    initial_accs = []
    best_accs = []
    hiddens = []

    for results in input:
        for key in results.keys():
            result = results[key]['results']
            initial_accs.append(result['accuracies'][0])
            best_accs.append(max(result['accuracies']))
            hiddens.append(key)

        ax.scatter(initial_accs, best_accs)
        ax.plot(np.unique(initial_accs), np.poly1d(np.polyfit(initial_accs, best_accs, 1))(np.unique(initial_accs)))

    if legend is not None:
        ax.legend(legend)


def good_net(h):
    for i in range(len(h)-1):
        if h[i] < h[i+1]:
            return False

    return True


plot_training_curves(results)
# plot_best_vs_first(results)

plot_best_vs_first(*split, legend=files)

# plot_best_vs_first(split[1])
# plot_training_curves(split[1])

plt.show()
