import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from utils.plot_utils import *

from utils.results_utils import ExperimentResults

data = {
    'bostonHousing':   {'dim':13,  'data_size':430},
    'concrete': {'dim':8,   'data_size':875},
    'kin8nm':   {'dim':8,   'data_size':652},
    'naval':    {'dim':8,   'data_size':6963},
    'power':    {'dim':16,  'data_size':10143},
    'protein':  {'dim':9,   'data_size':38870},
    'wine-quality-red':     {'dim':11,  'data_size':1359},
    'yacht':    {'dim':6,   'data_size':261}
}


def do_plots(results_dir):
    # thresholds = [[1], [0.002, 0.005], [1, 0.2, 0.2]]  # Hyperprior

    # thresholds = [[1], [1, 0.5], [1, 0.5, 0.5]] # Fixed Prior

    thresholds = [[0.3], [0.3, 1e-1], [0.3, 1e-1, 1e-1], [0.3, 1e-1, 1e-1, 1e-1], [0.3, 1e-2, 1e-2, 1e-2, 1e-2]]
    data_set = results_dir.split('/')[-2]

    def plot_all_pruning(layer_groups):

        fig, ax = plt.figure(figsize=(10, 5))

        plt.title('Comparison of total number of units vs active units')


    # def plot_KL_pruning(results):
    #     '''
    #     Note all results must have the same number of layers in them
    #     '''
    #
    #     layers = len(results[0]['hidden_sizes']) - 3
    #     prune_fig, prune_axs = plt.subplots(max(2, layers + 1), 2, figsize=(10, 2.5 * (layers + 1)))
    #
    #     prune_fig.suptitle(f'KL pruning behaviour for {layers} layers', fontsize=16)
    #
    #     for i in range(layers):
    #         prune_axs[i][0].set_title(f'Layer {i} active weights')
    #         prune_axs[i][0].set_xlabel('Layer size')
    #     prune_axs[layers][0].set_title('Total active weights')
    #     prune_axs[i][0].set_xlabel('Layer size')
    #
    #     # prune_axs[0][1].set_title('Final RMSE')
    #     prune_axs[0][1].set_title('Final Accuracy')
    #     prune_axs[1][1].set_title('Final Test Log Likelihood')
    #     # prune_axs[2][1].set_title('Layer 3 active weights')
    #     # prune_axs[3][1].set_title('Total active weights')
    #
    #     for i, r in enumerate(results):
    #         point = r['hidden_sizes'][1]
    #
    #         active_neurons = [
    #             sum(neuron_kl > thresholds[layers - 1][layer] for neuron_kl in r['results']['KL_pruning'][layer]) for
    #             layer in range(layers)]
    #
    #         final_rmse = sum(r['results']['test_rmse_true'][-20:]) / 20
    #         # final_rmse = sum(r['results']['test_acc'][-20:]) / 20
    #         final_test_loglik = sum(r['results']['test_ll_true'][-20:]) / 20
    #
    #         for i in range(layers):
    #             prune_axs[i][0].scatter(point, active_neurons[i])
    #         prune_axs[layers][0].scatter(point, sum(active_neurons))
    #
    #         prune_axs[0][1].scatter(point, final_rmse)
    #         prune_axs[1][1].scatter(point, final_test_loglik)
    #         # prune_axs[2][0].scatter(point, active_3)
    #         # prune_axs[3][0].scatter(point, active_1 + active_2 + active_3)
    #
    #         plt.tight_layout(rect=[0, 0.03, 1.0, 0.95])
    #         plt.savefig(os.path.join(results_dir, 'figs', f'pruning_kl_{layers}_layers.png'))

    # def plot_KL_pruning(results):
    #     '''
    #     Note all results must have the same number of layers in them
    #     '''
    #
    #     layers = len(results[0]['hidden_sizes']) - 3
    #     prune_fig, prune_axs = plt.subplots(2, 1, figsize=full_width_square)
    #     metric_fig, metric_axs = plt.subplots(2, 1, figsize=full_width_square)
    #
    #     # prune_fig.suptitle(f'KL pruning behaviour for {layers} layers', fontsize=16)
    #
    #     for i in range(2):
    #         prune_axs[i].set_xlabel('Layer size')
    #         metric_axs[i].set_xlabel('Layer size')
    #
    #     prune_axs[0].set_ylabel('Active neurons in each layer')
    #     prune_axs[1].set_ylabel('Total active neurons')
    #
    #     # prune_axs[0][1].set_title('Final RMSE')
    #     metric_axs[0].set_ylabel('Final RMSE')
    #     metric_axs[1].set_ylabel('Final Test Log Likelihood')
    #     # prune_axs[2][1].set_title('Layer 3 active weights')
    #     # prune_axs[3][1].set_title('Total active weights')
    #
    #     points = []
    #     active_weights = []
    #     active_weights_layers = [[] for _ in range(layers)]
    #     final_rmse = []
    #     final_test_loglik = []
    #
    #     for i, r in enumerate(results):
    #         points.append(r['hidden_sizes'][1])
    #
    #         active_neurons = [
    #             sum(neuron_kl > thresholds[layers - 1][layer] for neuron_kl in r['results']['KL_pruning'][layer]) for
    #             layer in range(layers)]
    #
    #         final_rmse.append(sum(r['results']['test_rmse_true'][-20:]) / 20)
    #         final_test_loglik.append(sum(r['results']['test_ll_true'][-20:]) / 20)
    #
    #         for i in range(layers):
    #             active_weights_layers[i].append(active_neurons[i])
    #
    #         active_weights.append(sum(active_neurons))
    #
    #     prune_axs[1].scatter(points, active_weights, c='tab:blue')
    #     for i in range(layers):
    #         prune_axs[0].scatter(points, active_weights_layers[i], label=f'layer {i+1}')
    #     prune_axs[0].legend()
    #
    #     metric_axs[0].scatter(points, final_rmse, color='tab:blue')
    #     metric_axs[1].scatter(points, final_test_loglik, color='tab:blue')
    #
    #     prune_fig.tight_layout() #rect=[0, 0.03, 1.0, 0.95]
    #     metric_fig.tight_layout()
    #     savefig_handle(prune_fig, os.path.join(results_dir, 'figs', f'pruning_kl_{layers}_layers'))
    #     savefig_handle(metric_fig, os.path.join(results_dir, 'figs', f'metircs_{layers}_layers'))
    #
    #     os.makedirs(os.path.join(final_thesis_dir, data_set), exist_ok=True)
    #     savefig_handle(prune_fig, os.path.join(final_thesis_dir, data_set, f'pruning_kl_{layers}_layers'), pdf=True)
    #     savefig_handle(metric_fig, os.path.join(final_thesis_dir, data_set, f'metircs_{layers}_layers'), pdf=True)


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


    def plot_kl_pruning_groups(layer_groups):

        groups = len(layer_groups)

        # all_pruning_fig, all_pruning_ax = plt.subplots(1,1,figsize=(5.6, 2.8))
        # all_rmse_fig, all_rmse_ax = plt.subplots(1,1,figsize=(5.6, 2.8))
        # all_log_lik_fig, all_log_lik_ax = plt.subplots(1,1,figsize=(5.6, 2.8))

        # data_set = results_dir.split('/')[2]
        # input_size = data[data_set]['dim']

        combined_fig, combined_ax = plt.subplots(3, 1, sharex=True, figsize=(text_width, text_height/1.2))

        combined_ax[2].set_xscale('log')

        for i, group in enumerate(layer_groups):

            results = layer_groups[group]
            '''
            Note all results must have the same number of layers in them
            '''

            layers = len(results[0]['hidden_sizes']) - 3
            prune_fig, prune_axs = plt.subplots(2, 1, figsize=full_width_square)
            metric_fig, metric_axs = plt.subplots(2, 1, figsize=full_width_square)

            # prune_fig.suptitle(f'KL pruning behaviour for {layers} layers', fontsize=16)

            for i in range(2):
                prune_axs[i].set_xlabel('Layer size')
                metric_axs[i].set_xlabel('Layer size')

            prune_axs[0].set_ylabel('Active neurons in each layer')
            prune_axs[1].set_ylabel('Total active neurons')

            # prune_axs[0][1].set_title('Final RMSE')
            metric_axs[0].set_ylabel('Final RMSE')
            metric_axs[1].set_ylabel('Final Test Log Likelihood')
            # prune_axs[2][1].set_title('Layer 3 active weights')
            # prune_axs[3][1].set_title('Total active weights')

            points = []
            active_weights = []
            active_weights_layers = [[] for _ in range(layers)]
            final_rmse = []
            final_test_loglik = []

            for i, r in enumerate(results):

                # net_shape = [input_size] + r['hidden_sizes'][1:-2] + [1]
                #
                # weights = 0
                # for i in range(len(net_shape)-1):
                #     weights += (net_shape[i] * net_shape[i+1])

                points.append(r['hidden_sizes'][1])
                # points.append(weights)

                active_neurons = [
                    sum(neuron_kl > thresholds[layers - 1][layer] for neuron_kl in r['results']['KL_pruning'][layer])
                    for layer
                    in range(layers)
                ]

                # active_neurons = [
                #     sum(neuron_kl for neuron_kl in r['results']['KL_pruning'][layer])
                #     for layer
                #     in range(layers)
                # ]

                final_rmse.append(sum(r['results']['test_rmse_true'][-20:]) / 20)
                final_test_loglik.append(sum(r['results']['test_ll_true'][-20:]) / 20)

                for i in range(layers):
                    active_weights_layers[i].append(active_neurons[i])

                active_weights.append(sum(active_neurons))


            prune_axs[1].scatter(points, active_weights, c='tab:blue')
            for i in range(layers):
                prune_axs[0].scatter(points, active_weights_layers[i], label=f'layer {i + 1}')
            prune_axs[0].legend()

            metric_axs[0].scatter(points, final_rmse, color='tab:blue')
            metric_axs[1].scatter(points, final_test_loglik, color='tab:blue')

            # all_pruning_ax.scatter(points, active_weights, label=f'{layers} layers')
            # all_rmse_ax.scatter(points, final_rmse, label=f'{layers} layers')
            # all_log_lik_ax.scatter(points, final_test_loglik, label=f'{layers} layers')


            combined_ax[0].scatter(points, active_weights, label=f'{layers} layers', c=colors[layers-1])
            combined_ax[2].scatter(points, final_rmse, label=f'{layers} layers', c=colors[layers-1])
            combined_ax[1].scatter(points, final_test_loglik, label=f'{layers} layers', c=colors[layers-1])

            lims = combined_ax[0].get_ylim()

            points = sorted(points)
            combined_ax[0].plot(points, [point * layers for point in points], c=colors[layers-1])
            # combined_ax[0].plot(points, points, c=colors[layers - 1])
            combined_ax[0].set_ylim(lims)

            prune_fig.tight_layout()  # rect=[0, 0.03, 1.0, 0.95]
            metric_fig.tight_layout()
            savefig_handle(prune_fig, os.path.join(results_dir, 'figs', f'pruning_kl_{layers}_layers'))
            savefig_handle(metric_fig, os.path.join(results_dir, 'figs', f'metircs_{layers}_layers'))

            os.makedirs(os.path.join(final_thesis_dir, data_set), exist_ok=True)
            savefig_handle(prune_fig, os.path.join(final_thesis_dir, data_set, f'pruning_kl_{layers}_layers'), pdf=True, png=False)
            savefig_handle(metric_fig, os.path.join(final_thesis_dir, data_set, f'metircs_{layers}_layers'), pdf=True, png=False)

        # all_pruning_ax.legend()
        # all_rmse_ax.legend()
        # all_log_lik_ax.legend()
        #
        # all_pruning_ax.set_xlabel('Layer size')
        # all_rmse_ax.set_xlabel('Layer size')
        # all_log_lik_ax.set_xlabel('Layer size')
        #
        # all_pruning_ax.set_ylabel('Total active neurons')
        # all_rmse_ax.set_ylabel('Final RMSE')
        # all_log_lik_ax.set_ylabel('Final Log Likelihood')
        #
        # all_pruning_ax.set_xscale('log')
        # all_rmse_ax.set_xscale('log')
        # all_log_lik_ax.set_xscale('log')
        #
        # all_pruning_ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        #
        # all_pruning_fig.tight_layout()
        # all_rmse_fig.tight_layout()
        # all_log_lik_fig.tight_layout()

        combined_ax[0].legend()
        combined_ax[2].legend()
        combined_ax[1].legend()

        combined_ax[2].set_xlabel('Layer size')

        combined_ax[0].set_ylabel('Total active neurons')
        combined_ax[2].set_ylabel('Final RMSE')
        combined_ax[1].set_ylabel('Final Log Likelihood')

        combined_ax[0].yaxis.set_major_locator(MaxNLocator(integer=True))

        combined_fig.tight_layout()


        # savefig_handle(all_pruning_fig, os.path.join(results_dir, 'figs', f'pruning_kl'))
        # savefig_handle(all_rmse_fig, os.path.join(results_dir, 'figs', f'pruning_rmse'))
        # savefig_handle(all_log_lik_fig, os.path.join(results_dir, 'figs', f'pruning_log_likelihood'))

        savefig_handle(combined_fig, os.path.join(results_dir, 'figs', f'pruning_combined'))

        # savefig_handle(all_pruning_fig, os.path.join(final_thesis_dir, data_set, f'pruning_kl'), pdf=True, png=False)
        # savefig_handle(all_rmse_fig, os.path.join(final_thesis_dir, data_set, f'pruning_rmse'), pdf=True, png=False)
        # savefig_handle(all_log_lik_fig, os.path.join(final_thesis_dir, data_set, f'pruning_log_likelihood'), pdf=True, png=False)

        savefig_handle(combined_fig, os.path.join(final_thesis_dir, data_set, f'pruning_combined'), pdf=True, png=False)

        plt.close('all')

    os.makedirs(os.path.join(results_dir, 'figs'), exist_ok=True)

    results = ExperimentResults(results_dir)

    layer_groups = results.group_results(lambda x: len(x['hidden_sizes']) - 3)

    plot_kl_pruning_groups(layer_groups)

    # for group in layer_groups: plot_KL_pruning(layer_groups[group])
    # plot_metric_actual_grouped(layer_groups, 'test_rmse_true')
    # plot_metric_actual_grouped(layer_groups, 'test_ll_true')
    # plot_metric_vs_metric(layer_groups, 'test_rmse_true', 'test_ll_true')


    # from utils.plot_utils import plot_KL_pruning_post
    # for group in layer_groups:
    #     for result in layer_groups[group]:
    #         plot_KL_pruning_post(result['results']['KL_pruning'], os.path.join(results_dir, 'figs'), str(result['hidden_sizes']))


    # for step in [0] + list(range(1000, len(layer_groups[1][0]['results']['test_ll_true']), 1000)):
    #     plot_optimisation_step_correlation_grouped(layer_groups, 'test_rmse_true', step)
    #     plot_optimisation_step_correlation_grouped(layer_groups, 'test_ll_true', step)

    # for group in layer_groups: plot_KL_pruning(layer_groups[group])
    # plot_metric_actual_grouped(layer_groups, 'test_acc')
    # plot_metric_actual_grouped(layer_groups, 'test_ll_true')
    # plot_metric_vs_metric(layer_groups, 'test_acc', 'test_ll_true')
    # for step in [0] + list(range(1000, len(layer_groups[0][0]['results']['test_ll_true']), 1000)):
    #     plot_optimisation_step_correlation_grouped(layer_groups, 'test_acc', step)
    #     plot_optimisation_step_correlation_grouped(layer_groups, 'test_ll_true', step)



    # plt.show()
    plt.close('all')

final_thesis_dir = '/home/mjhutchinson/Documents/University/4th Year/4th Year Project/Final Thesis/Thesis-LaTeX/Chapter5/Figs'

dirs_regression = [
        '../remote_logs_clean/bostonHousing/weight_pruning_hyperprior3',
        '../remote_logs_clean/concrete/weight_pruning_hyperprior3',
        '../remote_logs_clean/energy/weight_pruning_hyperprior3',
        '../remote_logs_clean/kin8nm/weight_pruning_hyperprior3',
        # '../remote_logs_clean/power-plant/weight_pruning_hyperprior3'
        # '../remote_logs_clean/wine-quality-red/weight_pruning_hyperprior3',
        # '../remote_logs_clean/protein-tertiary-structure/weight_pruning_hyperprior3',
        # '../remote_logs_clean/yacht/weight_pruning_hyperprior3',
]

dirs_classification = [
    # '../remote_logs_clean/mnist/weight_pruning_hyperprior', # [0.01]
]

for dir in dirs_regression:
    print(dir)
    do_plots(dir)
