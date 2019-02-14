import time
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os
import pickle

from utils.plot_utils import plot_KL_pruning, plot_SNP_pruning


def test_model_regression(model, data_gen, epochs, batch_size=100, log_freq=1, results_dir='./results', name_prefix=None,
                          accuracy_plots=True, KL_pruning_plots=True, SNR_pruning_plots=True, verbose=True):

    if name_prefix is None:
        name = f'{model}'
    else:
        name = f'{name_prefix}_{model}'

    result_file = f'{results_dir}/{name}.pkl'

    log_dir = f'{results_dir}/logs/{name}'

    if os.path.exists(result_file):
        print(f'{model} already exists, skipping')
        return pickle.load(open(result_file, 'rb'))['results']

    x_train, y_train, x_test, y_test = data_gen.get_data()
    y_sigma = float(data_gen.y_sigma)
    log_y_sigma = float(np.log(y_sigma))

    summary_writer = tf.summary.FileWriter(log_dir, graph=model.sess.graph)
    fig_dir = f'{log_dir}/figs'
    os.mkdir(fig_dir)

    train_elbos = []
    test_lls = []
    test_rmses = []
    train_lls = []
    train_kls = []
    noise_sigmas = []

    test_lls_true = []
    test_rmses_true = []
    train_lls_true = []
    noise_sigmas_true = []

    if verbose:
        test_acc, test_ll = model.accuracy(x_test, y_test, batch_size=batch_size)
        print(f'Initial test log likelihood: {test_ll:8.4f}, test accuracy: {test_acc:8.4f}')

    total_train_time = 0


    for epoch in range(epochs):

        t = time.time()
        train_elbo, train_kl, train_ll, optimisation_steps = model.train_one(x_train, y_train, batch_size=batch_size)
        train_time = time.time()-t

        total_train_time += train_time

        test_ll, test_rmse = model.accuracy(x_test, y_test, batch_size=batch_size)
        test_time = time.time() - train_time - t

        noise_sigma = model.sess.run(model.output_sigma)

        train_elbos.append(train_elbo)
        test_lls.append(test_ll)
        test_rmses.append(test_rmse)
        noise_sigmas.append(noise_sigma)
        train_lls.append(train_ll)
        train_kls.append(train_kl)

        test_ll_true = test_ll - log_y_sigma
        train_ll_true = train_ll - log_y_sigma
        test_rmse_true = test_rmse * y_sigma
        noise_sigma_true = noise_sigma * y_sigma

        test_lls_true.append(test_ll_true)
        train_lls_true.append(train_ll_true)
        test_rmses_true.append(test_rmse_true)
        noise_sigmas_true.append(noise_sigma_true)

        summary = model.log_metrics(train_elbo, train_ll, train_kl, test_ll, test_rmse, train_ll_true, test_ll_true, test_rmse_true, noise_sigma_true)
        summary_writer.add_summary(summary, epoch)

        if epoch % log_freq == 0 & verbose:
            print(f'\rOptimisation step {epoch:4.0f}, \t ELBO: {train_elbo:10.4f}, \t KL term: {train_kl:10.4f}, \t train log likelihood term: {train_ll_true:8.4f}, \t test log likelihood: {test_ll_true:8.4f}, \t test auxiliary: {test_rmse_true:8.4f}, \t noise sigma: {noise_sigma_true:8.4f}, \t train time: {train_time:6.4f}, \t test time: {test_time:6.4f}')

        if epoch % (10 * log_freq) == 0:
            # Plot predictions vs actual plots
            if accuracy_plots:
                predictions_train = np.mean(model.prediction(x_train, batch_size=batch_size), 0)
                predictions_test = np.mean(model.prediction(x_test, batch_size=batch_size), 0)
                plt.figure()
                plt.scatter(y_train, predictions_train)
                plt.scatter(y_test, predictions_test)
                plt.legend(['Train', 'Test'])
                plt.xlabel('actuals')
                plt.ylabel('predictions')
                plt.title(f'epcoh {epoch}')
                plt.plot([min(y_train), max(y_train)], [min(y_train), max(y_train)], color='r')
                plt.savefig(f'{fig_dir}/predictions_{epoch}.png')
                plt.close()

            if KL_pruning_plots:
                plot_KL_pruning(model, fig_dir, epoch)

            if SNR_pruning_plots:
                plot_SNP_pruning(model, fig_dir, epoch)


    summary_writer.close()

    result = {'elbo': train_elbos,
              'test_ll': test_lls,
              'test_rmse': test_rmses,
              'noise_sigma': noise_sigmas,
              'train_ll': train_lls,
              'train_kl': train_kls,
              'train_ll_true': train_lls_true,
              'test_ll_true': test_lls_true,
              'test_rmse_true': test_rmses_true,
              'noise_sigma_true': noise_sigma_true,
              'train_time': total_train_time}

    if KL_pruning_plots:
        pruning_measure = [weight.pruning_from_KL() for weight in model.W]
        pruning_measure = model.sess.run(pruning_measure)
        result['KL_pruning'] = pruning_measure

    if SNR_pruning_plots:
        pruning_measure = [weight.pruning_from_SNR() for weight in model.W]
        pruning_measure = model.sess.run(pruning_measure)
        result['SNR_pruning'] = pruning_measure

    model_config = model.get_config()
    train_config = {'batch_size': batch_size, 'epochs': epochs, 'results': result}
    output = {**model_config, **train_config, 'results': result}

    with open(result_file, 'wb') as h:
        pickle.dump(output, h)

    return result


def test_model_classification(model, data_gen, epochs, batch_size=100, log_freq=1, results_dir='./results',
                              name_prefix=None, accuracy_plots=True, KL_pruning_plots=True, SNR_pruning_plots=True,
                              verbose=True):

    if name_prefix is None:
        name = f'{model}'
    else:
        name = f'{name_prefix}_{model}'

    result_file = f'{results_dir}/{name}.pkl'

    log_dir = f'{results_dir}/logs/{name}'

    if os.path.exists(result_file):
        print(f'{model} already exists, skipping')
        return pickle.load(open(result_file, 'rb'))['results']

    x_train, y_train, x_test, y_test = data_gen.get_data()
    y_sigma = float(data_gen.y_sigma)
    log_y_sigma = float(np.log(y_sigma))

    summary_writer = tf.summary.FileWriter(log_dir, graph=model.sess.graph)
    fig_dir = f'{log_dir}/figs'
    os.mkdir(fig_dir)

    train_elbos = []
    test_lls = []
    test_accs = []
    train_lls = []
    train_kls = []

    test_lls_true = []
    train_lls_true = []

    times = []

    if verbose:
        test_acc, test_ll = model.accuracy(x_test, y_test, batch_size=batch_size)
        print(f'Initial test log likelihood: {test_ll:8.4f}, test acc: {test_acc:8.4f}')

    total_train_time = 0

    last_log = 0
    last_plot = 0

    for epoch in range(epochs):

        t = time.time()
        train_elbo, train_kl, train_ll = model.train_one(x_train, y_train, batch_size=batch_size)
        train_time = time.time()-t

        total_train_time += train_time

        test_acc, test_ll = model.accuracy(x_test, y_test, batch_size=batch_size)
        test_time = time.time() - train_time - t

        train_elbos.append(train_elbo)
        test_lls.append(test_ll)
        test_accs.append(test_acc)
        train_lls.append(train_ll)
        train_kls.append(train_kl)

        test_ll_true = test_ll - log_y_sigma
        train_ll_true = train_ll - log_y_sigma

        test_lls_true.append(test_ll_true)
        train_lls_true.append(train_ll_true)
        times.append(total_train_time)

        summary = model.log_metrics(train_elbo, train_ll, train_kl, test_ll, test_acc, train_ll_true, test_ll_true)
        summary_writer.add_summary(summary, epoch)

        if (epoch % log_freq == 0) & verbose:
            print(f'\rOptimisation step {epoch:4.0f}, \t ELBO: {train_elbo:10.4f}, \t KL term: {train_kl:10.4f}, \t train log likelihood term: {train_ll_true:8.4f}, \t test log likelihood: {test_ll_true:8.4f}, \t test accuracy: {test_acc:8.4f}, \t train time: {train_time:6.4f}, \t test time: {test_time:6.4f}')

        if epoch % (log_freq * 10) == 0:
            # Plot predictions vs actual plots

            # if accuracy_plots:
            #     predictions_train = np.mean(model.prediction(x_train, batch_size=batch_size), 0)
            #     predictions_test = np.mean(model.prediction(x_test, batch_size=batch_size), 0)
            #     plt.figure()
            #     plt.scatter(y_train, predictions_train)
            #     plt.scatter(y_test, predictions_test)
            #     plt.legend(['Train', 'Test'])
            #     plt.xlabel('actuals')
            #     plt.ylabel('predictions')
            #     plt.title(f'epcoh {epoch}')
            #     plt.plot([min(y_train), max(y_train)], [min(y_train), max(y_train)], color='r')
            #     plt.savefig(f'{fig_dir}/predictions_{epoch}.png')
            #     plt.close()

            if KL_pruning_plots:
                plot_KL_pruning(model, fig_dir, epoch)

            if SNR_pruning_plots:
                plot_SNP_pruning(model, fig_dir, epoch)


    summary_writer.close()

    result = {'elbo': train_elbos,
              'test_ll': test_lls,
              'test_acc': test_accs,
              'train_ll': train_lls,
              'train_kl': train_kls,
              'train_ll_true': train_lls_true,
              'test_ll_true': test_lls_true,
              'train_time': train_time}

    if KL_pruning_plots:
        pruning_measure = [weight.pruning_from_KL() for weight in model.W]
        pruning_measure = model.sess.run(pruning_measure)
        result['KL_pruning'] = pruning_measure

    if SNR_pruning_plots:
        pruning_measure = [weight.pruning_from_SNR() for weight in model.W]
        pruning_measure = model.sess.run(pruning_measure)
        result['SNR_pruning'] = pruning_measure

    model_config = model.get_config()
    train_config = {'batch_size': batch_size, 'epochs': epochs, 'results': result}
    output = {**model_config, **train_config, 'results': result}

    with open(result_file, 'wb') as h:
        pickle.dump(output, h)

    return result


def test_model_classification_optim_steps(model, data_gen, optimisation_steps, batch_size=100, log_freq=1, results_dir='./results',
                              name_prefix=None, accuracy_plots=True, KL_pruning_plots=True, SNR_pruning_plots=True,
                              verbose=True):

    if name_prefix is None:
        name = f'{model}'
    else:
        name = f'{name_prefix}_{model}'

    result_file = f'{results_dir}/{name}.pkl'

    log_dir = f'{results_dir}/logs/{name}'

    if os.path.exists(result_file):
        print(f'{model} already exists, skipping')
        return pickle.load(open(result_file, 'rb'))['results']

    x_train, y_train, x_test, y_test = data_gen.get_data()
    y_sigma = float(data_gen.y_sigma)
    log_y_sigma = float(np.log(y_sigma))

    summary_writer = tf.summary.FileWriter(log_dir, graph=model.sess.graph)
    fig_dir = f'{log_dir}/figs'
    os.mkdir(fig_dir)

    train_elbos = []
    test_lls = []
    test_accs = []
    train_lls = []
    train_kls = []

    test_lls_true = []
    train_lls_true = []

    if verbose:
        test_acc, test_ll = model.accuracy(x_test, y_test, batch_size=batch_size)
        print(f'Initial test log likelihood: {test_ll:8.4f}, test acc: {test_acc:8.4f}')

    total_train_time = 0

    for optimisation_step in range(optimisation_steps):

        t = time.time()
        train_elbo, train_kl, train_ll = model.train_one_optim_step(data_gen)
        train_time = time.time()-t

        total_train_time += train_time

        if (optimisation_step % log_freq == 0) & verbose:
            test_acc, test_ll = model.accuracy(x_test, y_test, batch_size=batch_size)
            test_time = time.time() - train_time - t

            train_elbos.append(train_elbo)
            test_lls.append(test_ll)
            test_accs.append(test_acc)
            train_lls.append(train_ll)
            train_kls.append(train_kl)

            test_ll_true = test_ll - log_y_sigma
            train_ll_true = train_ll - log_y_sigma

            test_lls_true.append(test_ll_true)
            train_lls_true.append(train_ll_true)

            summary = model.log_metrics(train_elbo, train_ll, train_kl, test_ll, test_acc, train_ll_true, test_ll_true)
            summary_writer.add_summary(summary, optimisation_step)

            print(f'\rOptimisation step {optimisation_step:4.0f}, \t ELBO: {train_elbo:10.4f}, \t KL term: {train_kl:10.4f}, \t train log likelihood term: {train_ll_true:8.4f}, \t test log likelihood: {test_ll_true:8.4f}, \t test accuracy: {test_acc:8.4f}, \t train time: {train_time:6.4f}, \t test time: {test_time:6.4f}')

        if optimisation_step % (log_freq * 10) == 0:
            # Plot predictions vs actual plots

            # if accuracy_plots:
            #     predictions_train = np.mean(model.prediction(x_train, batch_size=batch_size), 0)
            #     predictions_test = np.mean(model.prediction(x_test, batch_size=batch_size), 0)
            #     plt.figure()
            #     plt.scatter(y_train, predictions_train)
            #     plt.scatter(y_test, predictions_test)
            #     plt.legend(['Train', 'Test'])
            #     plt.xlabel('actuals')
            #     plt.ylabel('predictions')
            #     plt.title(f'epcoh {epoch}')
            #     plt.plot([min(y_train), max(y_train)], [min(y_train), max(y_train)], color='r')
            #     plt.savefig(f'{fig_dir}/predictions_{epoch}.png')
            #     plt.close()

            if KL_pruning_plots:
                plot_KL_pruning(model, fig_dir, optimisation_step)

            if SNR_pruning_plots:
                plot_SNP_pruning(model, fig_dir, optimisation_step)


    summary_writer.close()

    result = {'elbo': train_elbos,
              'test_ll': test_lls,
              'test_acc': test_accs,
              'train_ll': train_lls,
              'train_kl': train_kls,
              'train_ll_true': train_lls_true,
              'test_ll_true': test_lls_true,
              'train_time': total_train_time}

    if KL_pruning_plots:
        pruning_measure = [weight.pruning_from_KL() for weight in model.W]
        pruning_measure = model.sess.run(pruning_measure)
        result['KL_pruning'] = pruning_measure

    if SNR_pruning_plots:
        pruning_measure = [weight.pruning_from_SNR() for weight in model.W]
        pruning_measure = model.sess.run(pruning_measure)
        result['SNR_pruning'] = pruning_measure

    model_config = model.get_config()
    train_config = {'batch_size': batch_size, 'optimisation_steps': optimisation_steps, 'results': result}
    output = {**model_config, **train_config, 'results': result}

    with open(result_file, 'wb') as h:
        pickle.dump(output, h)

    return result



def get_activation(name):
    if name == 'relu':
        return tf.nn.relu
    elif name == 'tanh':
        return tf.nn.tanh
    elif name == 'sigmoid':
        return tf.nn.sigmoid
    else:
        raise ValueError(f'{name} is not a valid activation function')
