import time
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os
import pickle


def test_model_regression(model, data_gen, epochs, batch_size=100, log_freq=1, results_dir='./results', name_prefix=None, verbose=True):
    x_train, y_train, x_test, y_test = data_gen.get_data()

    if name_prefix is None:
        name = f'{model}'
    else:
        name = f'{name_prefix}_{model}'

    log_dir = f'{results_dir}/logs/{name}'

    summary_writer = tf.summary.FileWriter(log_dir, graph=model.sess.graph)
    fig_dir = f'{log_dir}/figs'
    os.mkdir(fig_dir)

    elbos = []
    test_lls = []
    rmses = []
    noise_sigmas = []
    train_lls = []
    train_kls = []

    if verbose:
        test_log_lik, test_rmse = model.accuracy(x_test, y_test, batch_size=batch_size)
        print(f'Initial test log likelihood: {test_log_lik:8.4f}, test rmse: {test_rmse:8.4f}')

    for epoch in range(epochs):

        t = time.time()
        train_elbo, train_kl, train_log_lik = model.train_one(x_train, y_train, batch_size=batch_size)
        train_time = time.time()-t
        test_log_lik, test_rmse = model.accuracy(x_test, y_test, batch_size=batch_size)
        test_time = time.time() - train_time - t

        sy = model.sess.run(model.output_sigma)

        elbos.append(train_elbo)
        test_lls.append(test_log_lik)
        rmses.append(test_rmse)
        noise_sigmas.append(sy)
        train_lls.append(train_log_lik)
        train_kls.append(train_kl)

        summary = model.log_metrics(train_elbo, train_log_lik, train_kl, test_log_lik, test_rmse)
        summary_writer.add_summary(summary, epoch)

        if (epoch % log_freq == 0) & verbose:
            print(f'\rEpoch {epoch:4.0f}, \t ELBO: {train_elbo:10.4f}, \t KL term: {train_kl:10.4f}, \t train log likelihood term: {train_log_lik:8.4f}, \t test log likelihood: {test_log_lik:8.4f}, \t test auxiliary: {test_rmse:8.4f}, \t noise sigma: {sy:8.4f}, \t train time: {train_time:6.4f}, \t test time: {test_time:6.4f}')

        if epoch % (log_freq * 10) == 0:
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
            plt.savefig(f'{fig_dir}/{epoch}.png')
            plt.close()

    summary_writer.close()

    result = {'elbo': elbos, 'test_ll': test_lls, 'test_rmse': rmses, 'noise_sigma': noise_sigmas, 'train_ll': train_lls, 'train_kl': train_kls}

    model_config = model.get_config()
    train_config = {'batch_size': batch_size, 'epochs': epochs, 'results': result}
    output = {**model_config, **train_config, 'results': result}

    result_file = f'{results_dir}/{name}.pkl'
    with open(result_file, 'wb') as h:
        pickle.dump(output, h)

    return result


def test_model_classification(model, data_gen, epochs, batch_size=100, log_freq=1, log_dir='logs'):
    x_train, y_train, x_test, y_test = data_gen.get_data()

    summary_writer = tf.summary.FileWriter(log_dir, graph=model.sess.graph)
    fig_dir = f'{log_dir}/figs'
    os.mkdir(fig_dir)

    costs = []
    test_ll = []
    accuracies = []
    train_ll = []
    train_kl = []

    for epoch in range(epochs):
        t = time.time()
        cost, kl, ll = model.train_one(x_train, y_train, batch_size=batch_size)
        train_time = time.time()-t
        accuracy, logloss = model.accuracy(x_test, y_test, batch_size=batch_size)
        test_time = time.time() - train_time - t

        costs.append(cost)
        test_ll.append(logloss)
        accuracies.append(accuracy)
        train_ll.append(ll)
        train_kl.append(kl)

        summary = model.log_metrics(cost, ll, kl, logloss, accuracy)
        summary_writer.add_summary(summary, epoch)

        if epoch % log_freq == 0:print(f'\rEpoch {epoch:4.0f}, cost: {cost:14.4f}, KL term: {kl:10.4f}, log likelihood part: {ll:10.4f}, accuracy: {accuracy:10.4f}, train time: {train_time:6.4f}, test time: {test_time:6.4f}')

        if epoch % (log_freq * 10) == 0:
            predictions = np.mean(model.prediction(x_train, batch_size=batch_size), 0)
            plt.scatter(y_train, predictions)
            plt.xlabel('actuals')
            plt.ylabel('predictions')
            # plt.plot([min(y_train), max(y_train)], [min(y_train), max(y_train)], color='r')
            plt.savefig(f'{fig_dir}/{epoch}.png')
            plt.close()

    summary_writer.close()

    return {'costs': costs, 'test_ll': test_ll, 'accuracies': accuracies, 'train_ll': train_ll,
            'train_kl': train_kl}


def get_activation(name):
    if name == 'relu':
        return tf.nn.relu
    elif name == 'tanh':
        return tf.nn.tanh
    elif name == 'sigmoid':
        return tf.nn.sigmoid
    else:
        raise ValueError(f'{name} is not a valid activation function')
