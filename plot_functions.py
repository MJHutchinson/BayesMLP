import matplotlib.pyplot as plt
import numpy as np


def plot_training_curves(input, val = 'accuracies', legend=None):
    _, ax = plt.subplots(1, 1)
    ax.set_xlabel('Epoch')
    ax.set_ylabel(val)
    ax.set_title(val)
    if legend is None:
        legend = []
    for results in input:
        result = results['results']
        ax.plot(result[val])
        legend.append(f'{results["hidden_sizes"]}_{results["learning_rate"]}_{results["prior_var"]}')

    ax.legend(legend)

def plot_training_curves_rv(input, legend=None, rolling_av_len=5):
    _, ax = plt.subplots(1, 1)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    if legend is None:
        legend = []
    for results in input:
        for key in results.keys():
            acc = results[key]['results']['accuracies']
            av_acc = [0] * (len(acc) - rolling_av_len)
            for i, _ in enumerate(av_acc):
                for j in range(rolling_av_len):
                    av_acc[i] += acc[i+j]/rolling_av_len
            ax.plot(av_acc)

        ax.legend(legend)



def plot_cost_curves(*input, legend=None, key='rmse'):
    _, ax = plt.subplots(1, 1)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Cost')
    legend = []
    for results in input:
        result = results['results']
        ax.plot(result['costs'])
        legend.append(key)

    ax.legend(legend)


def plot_min_vs_first(input, val = 'costs', legend=None):
    _, ax = plt.subplots(1, 1)
    ax.set_xlabel(f'First epoch {val}')
    ax.set_ylabel(f'Minimum {val}')

    initial_accs = []
    best_accs = []

    for result in input:

        r = result['results'][val]
        initial_accs.append(r[0])
        best_accs.append(min(r))

    ax.scatter(initial_accs, best_accs)
    ax.plot(np.unique(initial_accs), np.poly1d(np.polyfit(initial_accs, best_accs, 1))(np.unique(initial_accs)))

    if legend is not None:
        ax.legend(legend)

def rank_best_value(input, n=10, value = 'accuracies', minimum=False):
    print(f'{"Minimum" if minimum else "Maximum"} {value} (limited to {n})')
    for results in input:
        pairs = []
        for i, key in enumerate(results.keys()):
            pairs.append((key, min(results[key]['results'][value]) if minimum else max(results[key]['results'][value])))

        pairs = sorted(pairs, key = lambda t: t[1], reverse=not minimum)

        for i, pair in enumerate(pairs):
            if i<10:
                print(f'{pair[0]}: {value}: {pair[1]}')

    print('\n')


def rank_final_value(*input, n=10, value = 'accuracies', minimum=False):
    print(f'{"Minimum" if minimum else "Maximum"} final {value} (limited to {n})')
    for results in input:
        pairs = []
        for i, key in enumerate(results.keys()):
            pairs.append((key, results[key]['results'][value][-1]))

        pairs = sorted(pairs, key = lambda t: t[1], reverse=not minimum)

        for i, pair in enumerate(pairs):
            if i<10:
                print(f'{pair[0]}: {value}: {pair[1]}')