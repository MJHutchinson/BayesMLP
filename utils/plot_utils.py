import matplotlib.pyplot as plt
import matplotlib
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import rc

# plt.style.use('bmh')

plt.rcParams["font.family"] = "serif"
plt.rcParams["text.usetex"] = True

rc('text', usetex=True)
rc('font', size=9)
rc('lines', linewidth=0.5)
rc('lines', markersize=6)
rc('scatter', marker='+')
rc('axes', grid=True)
rc('axes', axisbelow=True)
rc('patch', edgecolor='black')
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})

cmap = plt.get_cmap('Spectral')
cm = lambda x: cmap(1 - x)

rc('text', usetex=True)
rc('font', size=10)
rc('lines', linewidth=1)
# rc('lines', markersize=1)
# rc('scatter', marker='|')
rc('axes', grid=True)
rc('patch', edgecolor='black')
# rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})

cmap = plt.get_cmap('winter')
cm = lambda x: cmap(1 - x)

pdf_params = {
    'format': 'pdf',
    'bbox_inches': 'tight',
    # 'dpi': 1000
}
png_params = {
    'format': 'png',
    'bbox_inches': 'tight',
    # 'dpi': 1000
}
svg_params = {
    'format': 'svg',
    'bbox_inches': 'tight',
    # 'dpi': 1000
}

colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']


half_width_square = (2.8, 2.8)
full_width_square = (5.6, 5.6)

text_width = 5.8
text_height = 9.1


def savefig(fig_name, pdf=False, svg=False, png=True):
    if png: plt.savefig(fig_name + '.png', **png_params)
    if svg: plt.savefig(fig_name + '.svg', **svg_params)
    if pdf: plt.savefig(fig_name + '.pdf', **pdf_params)


def savefig_handle(fig, fig_name, pdf=False, svg=False, png=True):
    if png: fig.savefig(fig_name + '.png', **png_params)
    if svg: fig.savefig(fig_name + '.svg', **svg_params)
    if pdf: fig.savefig(fig_name + '.pdf', **pdf_params)


def plot_training_curves(input, val='accuracies', rolling_av_len=20, legend=None, title=None, ax=None, xlabel=True):
    if ax==None:
        fig, ax = plt.subplots(1, 1)
    else:
        fig = None

    if xlabel:
        ax.set_xlabel('Epoch')

    if title is None:
        ax.set_ylabel(val)
        # ax.set_title(val)
    else:
        ax.set_ylabel(title)
        # ax.set_title(title)

    # if legend is None:
    #     legend = []
    for results in input:
        if rolling_av_len is None:
            result = results['results']
            if legend is None:
                ax.plot(result[val]) # , label=results['data_multiply']
            else:
                ax.plot(result[val], label=results[legend])  # ,
            # legend.append(f'{results["hidden_size"]} lr: {results["lr"]} prior width: {results["prior_var"]}')
        else:
            vals = result = results['results'][val]
            smoothed_vals = [0] * (len(vals) - rolling_av_len)
            for i, _ in enumerate(smoothed_vals):
                smoothed_vals[i] = sum(vals[i:i+rolling_av_len])/rolling_av_len

            if legend is None:
                ax.plot(smoothed_vals) # , label=results['data_multiply'] # Figure this out...
            else:
                ax.plot(smoothed_vals, label=results[legend])

    # if legend is not None:
    #     ax.legend(legend)

    return fig, ax


def plot_min_vs_i(input, i, val = 'costs', legend=None):
    _, ax = plt.subplots(1, 1)
    ax.set_xlabel(f'Epoch {i+1} {val}')
    ax.set_ylabel(f'Minimum {val}')
    ax.set_title(f'Plot of epoch {i+1} {val} vs minimum {val}')

    initial_accs = []
    best_accs = []

    for result in input:

        r = result['results'][val]
        initial_accs.append(r[i])
        best_accs.append(min(r))

    ax.scatter(initial_accs, best_accs)
    # ax.plot(np.unique(initial_accs), np.poly1d(np.polyfit(initial_accs, best_accs, 1))(np.unique(initial_accs)))

    if legend is not None:
        ax.legend(legend)


def plot_max_vs_i(input, i, val = 'costs', legend=None):
    _, ax = plt.subplots(1, 1)
    ax.set_xlabel(f'Epoch {i+1} {val}')
    ax.set_ylabel(f'Maximum {val}')
    ax.set_title(f'Plot of epoch {i+1} {val} vs maximum {val}')


    initial_accs = []
    best_accs = []
    legend = []

    for result in input:

        r = result['results'][val]
        initial_accs.append(r[i])
        best_accs.append(max(r))
        ax.scatter(r[i], max(r))
        legend.append(f'{result["hidden_size"]} lr: {result["learning_rate"]} prior width: {result["prior_var"]}')

    # ax.scatter(initial_accs, best_accs)
    # ax.plot(np.unique(initial_accs), np.poly1d(np.polyfit(initial_accs, best_accs, 1))(np.unique(initial_accs)))

    if legend is not None:
        ax.legend(legend)


def plot_last_vs_i(input, i, val = 'costs', legend=None):
    _, ax = plt.subplots(1, 1)
    ax.set_xlabel(f'{i} epoch {val}')
    ax.set_ylabel(f'Final epoch {val}')

    initial_accs = []
    best_accs = []

    for result in input:
        r = result['results'][val]
        initial_accs.append(r[0])
        best_accs.append(r[-1])

    ax.scatter(initial_accs, best_accs)
    ax.plot(np.unique(initial_accs), np.poly1d(np.polyfit(initial_accs, best_accs, 1))(np.unique(initial_accs)))

    if legend is not None:
        ax.legend(legend)


def plot_xy(x, y, x_lablel='', y_label='', legend=None):
    _, ax = plt.subplots(1, 1)
    ax.set_xlabel(x_lablel)
    ax.set_ylabel(y_label)

    ax.scatter(x, y)

    if legend is not None:
        ax.legend(legend)


def plot_dict(x_dict, y_dict, x_label=None, y_label=None, title=None, log_scale=False, use_legend=True, ax=None):
    if ax == None:
        fig, ax = plt.subplots(1, 1)
    else:
        fig = None
    if x_label is not None: ax.set_xlabel(x_label)
    if y_label is not None: ax.set_ylabel(y_label)

    if log_scale: ax.set_xscale('log')

    legend = list(x_dict.keys())

    try:
        legend = [int(l) for l in legend]
    except ValueError:
        try:
            legend = [float(l) for l in legend]
        except ValueError:
            pass

    legend = sorted(legend)

    for key in legend:
        if type(key) is not str:
            key = repr(key)
        ax.scatter(x_dict[key], y_dict[key])

    if use_legend:
        ax.legend(legend)

    if title is not None:
        ax.set_title(title)

    return fig, ax


def rank_best_value(input, n=10, value = 'accuracies', minimum=False):
    print(f'{"Minimum" if minimum else "Maximum"} {value} (limited to {n})')
    pairs = []
    for results in input:
        pairs.append((results['hidden_size'], min(results['results'][value]) if minimum else max(results['results'][value])))

    pairs = sorted(pairs, key = lambda t: t[1], reverse=not minimum)

    for i, pair in enumerate(pairs):
        if i<10:
            print(f'{pair[0]}: {value}: {pair[1]}')

    print('\n')


def rank_final_value(*input, n=10, value = 'accuracies', minimum=False):
    print(f'{"Minimum" if minimum else "Maximum"} final {value} (limited to {n})')
    for results in input:
        pairs = []
        for result in results:
            pairs.append((f'{result["hidden_sizes"]} lr: {result["learning_rate"]} prior width: {result["prior_var"]}', np.mean(result['results'][value][-20:])))

        pairs = sorted(pairs, key = lambda t: t[1], reverse=not minimum)

        for i, pair in enumerate(pairs):
            if i<10:
                print(f'{pair[0]}: {value}: {pair[1]}')


def plot_KL_pruning(model, fig_dir, epoch):
    # Plot cdf of 'pruning' based on KL
    pruning_measure = [weight.pruning_from_KL() for weight in model.W]
    pruning_measure = model.sess.run(pruning_measure)
    # pruning_measure = np.concatenate(pruning_measure)

    fig, axs = plt.subplots(len(pruning_measure), 1, figsize=(6.4, 2.4 * len(pruning_measure)))
    axs[0].set_title('Reverse CDF of weight pruning by KL')
    for i, (x, ax) in enumerate(zip(pruning_measure, axs)):
        ax.hist(x, bins=100, density=False, cumulative=-1, label=f'Layer {i}',
                histtype='step', alpha=1.0)
        ax.set_xlabel('Mean KL of weights')
        ax.set_ylabel('Cumulative density')
        ax.legend()
    plt.tight_layout()
    plt.savefig(f'{fig_dir}/pruning_KL_CDF_{epoch}.png')
    plt.close()

    fig, axs = plt.subplots(len(pruning_measure), 1, figsize=(6.4, 2.4 * len(pruning_measure)))
    axs[0].set_title('PDF of weight pruning by KL')
    for (x, ax) in zip(pruning_measure, axs):
        ax.hist(x, bins=100, density=False, cumulative=False, label=f'Layer {i}',
                histtype='step', alpha=1.0)

        mu = float(np.mean(x))
        std = float(np.std(x))
        ax.plot([mu + std, mu + std], list(ax.get_ylim()), label=f'Layer {i}: mu + std')
        active = np.sum(x > (mu + std))
        ax.text(0.25, 0.9 - 0.06 * i, f'layer {i}: KLs>mu+std: {active}', transform=ax.transAxes)

        ax.set_xlabel('Mean KL of weights')
        ax.set_ylabel('Density')
        ax.legend()
    plt.tight_layout()
    plt.savefig(f'{fig_dir}/pruning_KL_PDF_{epoch}.png')
    plt.close()

def plot_KL_pruning_post(pruning_measure, fig_dir, name):
    # Plot cdf of 'pruning' based on KL
    # Same as above, but for using after the fact

    # fig, axs = plt.subplots(len(pruning_measure), 1, figsize=(6.4, 2.4 * len(pruning_measure)))
    # axs[0].set_title('Reverse CDF of weight pruning by KL')
    # for i, (x, ax) in enumerate(zip(pruning_measure, axs)):
    #     ax.hist(x, bins=100, density=False, cumulative=-1, label=f'Layer {i}',
    #             histtype='step', alpha=1.0)
    #     ax.set_xlabel('Mean KL of weights')
    #     ax.set_ylabel('Cumulative density')
    #     ax.legend()
    # plt.tight_layout()
    # plt.savefig(f'{fig_dir}/{name}_pruning_KL_CDF_final.png')
    # plt.close()

    fig, axs = plt.subplots(len(pruning_measure), 1, figsize=(6.4, 2.4 * len(pruning_measure)))
    axs[0].set_title('PDF of weight pruning by KL')
    for i, (x, ax) in enumerate(zip(pruning_measure, axs)):
        ax.hist(np.log(x), bins=100, density=False, cumulative=False, label=f'Layer {i}',
                histtype='step', alpha=1.0)

        # mu = float(np.mean(x))
        # std = float(np.std(x))

        thresholds = [[0.5], [0.5, 0.05], [0.5, 0.05, 0.05], [0.5, 0.05, 0.05, 0.05], [0.5, 0.05, 0.05, 0.05, 0.05]]
        # print(len(pruning_measure), i)
        # print(len(thresholds), len(thresholds[len(pruning_measure)]))
        threshold = thresholds[len(pruning_measure)][i]
        ax.plot([threshold, threshold], list(ax.get_ylim()), label=f'Layer {i}: threshold')
        active = np.sum(x > threshold)
        ax.text(0.25, 0.9 - 0.06 * i, f'layer {i}: KLs>threshold: {active}', transform=ax.transAxes)

        ax.set_xlabel('Mean KL of weights')
        ax.set_ylabel('Density')
        ax.legend()
    plt.tight_layout()
    plt.savefig(f'{fig_dir}/{name}_pruning_KL_PDF_final.png')
    plt.close()


def plot_SNP_pruning(model, fig_dir, epoch):
    # Plot cdf of 'pruning' based on SNR
    pruning_measure = [weight.pruning_from_SNR() for weight in model.W]
    pruning_measure = model.sess.run(pruning_measure)
    # pruning_measure = np.concatenate(pruning_measure)

    fig, axs = plt.subplots(len(pruning_measure), 1, figsize=(6.4, 2.4 * len(pruning_measure)))
    axs[0].set_title('Reverse CDF of weight pruning by SNR')
    for i, (x, ax) in enumerate(zip(pruning_measure, axs)):
        ax.hist(x, bins=100, density=False, cumulative=-1, label=f'Layer {i}',
                histtype='step', alpha=1.0)
        ax.set_xlabel('SNR of weights')
        ax.set_ylabel('Cumulative density')
        ax.legend()
    plt.tight_layout()
    plt.savefig(f'{fig_dir}/pruning_SNR_CDF_{epoch}.png', )
    plt.close()

    fig, axs = plt.subplots(len(pruning_measure), 1, figsize=(6.4, 2.4 * len(pruning_measure)))
    axs[0].set_title('PDF of weight pruning by SNR')
    for (x, ax) in zip(pruning_measure, axs):
        ax.hist(x, bins=100, density=False, cumulative=False, label=f'Layer {i}',
                histtype='step', alpha=1.0)

        # mu = float(np.mean(x))
        # std = float(np.std(x))
        # ax.plot([mu + std, mu + std], list(ax.get_ylim()), label=f'Layer {i}: mu + std')
        # active = np.sum(x > (mu + std))
        # ax.text(0.25, 0.9 - 0.06 * i, f'layer {i}: KLs>mu+std: {active}', transform=ax.transAxes)

        ax.set_xlabel('SNR of weights')
        ax.set_ylabel('Density')
        ax.legend()
    plt.tight_layout()
    plt.savefig(f'{fig_dir}/pruning_SNR_PDF_{epoch}.png')
    plt.close()
