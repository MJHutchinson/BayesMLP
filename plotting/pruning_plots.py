import os

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "serif"
plt.rcParams["text.usetex"] = True

from utils.results_utils import ExperimentResults


# results_dir = '../remote_logs_clean/bostonHousing/weight_pruning_hyperprior'
# results_dir = '../remote_logs_clean/bostonHousing/weight_pruning_prior_1'

# results_dir = '../remote_logs_clean/wine-quality-red/weight_pruning_hyperprior'
# results_dir = '../remote_logs_clean/wine-quality-red/weight_pruning_prior_1'

# results_dir = '../remote_logs_clean/yacht/weight_pruning_hyperprior'
# results_dir = '../remote_logs_clean/yacht/weight_pruning_prior_1'

# results_dir = '../remote_logs_clean/protein-tertiary-structure/weight_pruning_hyperprior'
# results_dir = '../remote_logs_clean/protein-tertiary-structure/weight_pruning_prior_1'

# results_dir = '../remote_logs_clean/concrete/weight_pruning_hyperprior'
# results_dir = '../remote_logs_clean/concrete/weight_pruning_prior_1'
#
# results_dir = '../remote_logs_clean/energy/weight_pruning_hyperprior'
# results_dir = '../remote_logs_clean/energy/weight_pruning_prior_1'
#
# results_dir = '../remote_logs_clean/kin8nm/weight_pruning_hyperprior'
# results_dir = '../remote_logs_clean/kin8nm/weight_pruning_prior_1'
#
# results_dir = '../remote_logs_clean/naval-propulsion-plant/weight_pruning_hyperprior'
# results_dir = '../remote_logs_clean/naval-propulsion-plant/weight_pruning_prior_1'
#
# results_dir = '../remote_logs_clean/power-plant/weight_pruning_hyperprior'
# results_dir = '../remote_logs_clean/power-plant/weight_pruning_prior_1'

def do_plots(results_dir):
    os.makedirs(os.path.join(results_dir, 'figs'), exist_ok=True)

    results = ExperimentResults(results_dir)

    layer_groups = results.group_results(lambda x: len(x['hidden_sizes']) - 3)

    thresholds = [[1], [1, 0.5], [1, 0.2, 0.2]]  # Hyperprior

    # thresholds = [[1], [1, 0.5], [1, 0.5, 0.5]] # Fixed Prior

    def plot_KL_pruning(results):
        '''
        Note all results must have the same number of layers in them
        '''

        layers = len(results[0]['hidden_sizes']) - 3
        prune_fig, prune_axs = plt.subplots(max(2, layers + 1), 2, figsize=(10, 2.5 * (layers + 1)))

        prune_fig.suptitle(f'KL pruning behaviour for {layers} layers', fontsize=16)

        for i in range(layers):
            prune_axs[i][0].set_title(f'Layer {i} active weights')
            prune_axs[i][0].set_xlabel('Layer size')
        prune_axs[layers][0].set_title('Total active weights')
        prune_axs[i][0].set_xlabel('Layer size')

        prune_axs[0][1].set_title('Final RMSE')
        prune_axs[1][1].set_title('Final Test Log Likelihood')
        # prune_axs[2][1].set_title('Layer 3 active weights')
        # prune_axs[3][1].set_title('Total active weights')

        for i, r in enumerate(results):
            point = r['hidden_sizes'][1]

            active_neurons = [
                sum(neuron_kl > thresholds[layers - 1][layer] for neuron_kl in r['results']['KL_pruning'][layer]) for
                layer in range(layers)]

            final_rmse = sum(r['results']['test_rmse_true'][-20:]) / 20
            final_test_loglik = sum(r['results']['test_ll_true'][-20:]) / 20

            for i in range(layers):
                prune_axs[i][0].scatter(point, active_neurons[i])
            prune_axs[layers][0].scatter(point, sum(active_neurons))

            prune_axs[0][1].scatter(point, final_rmse)
            prune_axs[1][1].scatter(point, final_test_loglik)
            # prune_axs[2][0].scatter(point, active_3)
            # prune_axs[3][0].scatter(point, active_1 + active_2 + active_3)

            plt.tight_layout(rect=[0, 0.03, 1.0, 0.95])
            plt.savefig(os.path.join(results_dir, 'figs', f'pruning_kl_{layers}_layers.png'))

    # def plot_rankings(results, metric):

    def plot_metric_actual_grouped(groups, metric):

        fig, axs = plt.subplots(1, 1, figsize=(5, 2.5))

        for group in groups:

            results = groups[group]

            final_metrics = []
            layer_sizes = []

            for result in results:
                final_metric = result['results'][metric][-20:]
                final_metric = sum(final_metric) / len(final_metric)

                final_metrics.append(final_metric)
                layer_sizes.append(result['hidden_sizes'][1])

            plt.scatter(layer_sizes, final_metrics)

        plt.legend(groups.keys())
        plt.xlabel('Hidden layer size')
        plt.ylabel(metric.replace('_', ' '))

        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'figs', f'{metric}.png'))

    def plot_metric_vs_metric(groups, metric1, metric2):
        fig, axs = plt.subplots(1, 1, figsize=(5, 2.5))

        for group in groups:

            results = groups[group]

            final_metrics1 = []
            final_metrics2 = []

            for result in results:
                final_metric1 = result['results'][metric1][-20:]
                final_metric1 = sum(final_metric1) / len(final_metric1)

                final_metrics1.append(final_metric1)

                final_metric2 = result['results'][metric2][-20:]
                final_metric2 = sum(final_metric2) / len(final_metric2)

                final_metrics2.append(final_metric2)

            plt.scatter(final_metrics1, final_metrics2)

        plt.legend(groups.keys())
        plt.xlabel(metric1.replace('_', ' '))
        plt.ylabel(metric2.replace('_', ' '))

        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'figs', f'{metric1}-{metric2}.png'))


    def plot_optimisation_step_correlation_grouped(groups, metric, step):

        fig, axs = plt.subplots(1, 1, figsize=(5, 2.5))

        for group in groups:
            final_vals = []
            step_vals = []

            results = groups[group]
            for result in results:
                final_vals.append(result['results'][metric][-1])
                step_vals.append(result['results'][metric][step])

            plt.scatter(step_vals, final_vals,)

        plt.legend(groups.keys())
        plt.xlabel(f'{metric.replace("_", " ")} at step {step}')
        plt.ylabel(f'{metric.replace("_", " ")} at final step')
        plt.plot([min(final_vals), max(final_vals)], [min(final_vals), max(final_vals)])

        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'figs', f'{metric}_correlation_step_{step}.png'))


    for group in layer_groups: plot_KL_pruning(layer_groups[group])
    plot_metric_actual_grouped(layer_groups, 'test_rmse_true')
    plot_metric_actual_grouped(layer_groups, 'test_ll_true')
    plot_metric_vs_metric(layer_groups, 'test_rmse_true', 'test_ll_true')
    for step in [0] + list(range(1000, len(layer_groups[1][0]['results']['test_ll_true']), 1000)):
        plot_optimisation_step_correlation_grouped(layer_groups, 'test_rmse_true', step)
        plot_optimisation_step_correlation_grouped(layer_groups, 'test_ll_true', step)



    # plt.show()
    plt.close('all')


dirs = [
        # '../remote_logs_clean/bostonHousing/weight_pruning_hyperprior',
        # '../remote_logs_clean/bostonHousing/weight_pruning_prior_1',
        # '../remote_logs_clean/wine-quality-red/weight_pruning_hyperprior',
        # '../remote_logs_clean/wine-quality-red/weight_pruning_prior_1',
        # '../remote_logs_clean/yacht/weight_pruning_hyperprior',
        # '../remote_logs_clean/yacht/weight_pruning_prior_1',
        # '../remote_logs_clean/protein-tertiary-structure/weight_pruning_hyperprior',
        # '../remote_logs_clean/protein-tertiary-structure/weight_pruning_prior_1',
        # '../remote_logs_clean/concrete/weight_pruning_hyperprior',
        # '../remote_logs_clean/concrete/weight_pruning_prior_1',
        # '../remote_logs_clean/energy/weight_pruning_hyperprior',
        # '../remote_logs_clean/energy/weight_pruning_prior_1',
        # '../remote_logs_clean/kin8nm/weight_pruning_hyperprior',
        # '../remote_logs_clean/kin8nm/weight_pruning_prior_1',
        # '../remote_logs_clean/naval-propulsion-plant/weight_pruning_hyperprior',
        # '../remote_logs_clean/naval-propulsion-plant/weight_pruning_prior_1',
        '../remote_logs_clean/power-plant/weight_pruning_hyperprior',
        '../remote_logs_clean/power-plant/weight_pruning_prior_1'
]

for dir in dirs:
    print(dir)
    do_plots(dir)
