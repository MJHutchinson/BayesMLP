import time
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

def test_model_regression(model, data_gen, epochs, batch_size=100, log_freq=1, log_dir='logs'):
    x_train, y_train, x_test, y_test = data_gen.get_data()

    summary_writer = tf.summary.FileWriter(log_dir, graph=model.sess.graph)

    costs = []
    test_ll = []
    rmses = []
    noise_var = []
    train_ll = []
    train_kl = []

    for epoch in range(epochs):
        t = time.time()
        cost, kl, ll = model.train_one(x_train, y_train, batch_size=batch_size)
        train_time = time.time()-t
        logloss, rmse = model.accuracy(x_test, y_test, batch_size=batch_size)
        test_time = time.time() - train_time - t

        vy = model.sess.run(model.noise_var)

        costs.append(cost)
        test_ll.append(logloss)
        rmses.append(rmse)
        noise_var.append(vy)
        train_ll.append(ll)
        train_kl.append(kl)

        summary = model.log_metrics(cost, ll, kl, logloss, rmse)
        summary_writer.add_summary(summary, epoch)

        if epoch % log_freq == 0:print(f'\rEpoch {epoch:4.0f}, cost: {cost:14.4f}, KL term: {kl:10.4f}, train log likelihood term: {ll:10.4f}, test log likelihood: {logloss:10.4f}, test rmse: {rmse:10.4f}, log noise var: {vy:10f}, train time: {train_time:6.4f}, test time: {test_time:6.4f}')

    # predictions = np.mean(model.prediction(x_train, batch_size=batch_size), 0)
    # plt.scatter(y_train, predictions)
    # plt.xlabel('actuals')
    # plt.ylabel('predictions')
    # plt.plot([min(y_train), max(y_train)], [min(y_train), max(y_train)], color='r')
    # plt.show()

    summary_writer.close()

    return {'costs': costs, 'test_ll': test_ll, 'rmses': rmses, 'noise_sigma': noise_var, 'train_ll': train_ll, 'train_kl': train_kl}


def test_model_classification(model, data_gen, epochs, batch_size=100, log_freq=1):
    x_train, y_train, x_test, y_test = data_gen.get_data()

    costs = []
    accuracies = []

    for epoch in range(epochs):
        t = time.time()
        cost, kl, ll = model.train_one(x_train, y_train, batch_size=batch_size)
        train_time = time.time()-t
        accuracy = model.accuracy(x_test, y_test, batch_size=batch_size)
        test_time = time.time() - train_time - t

        costs.append(cost)
        accuracies.append(accuracy)
        if epoch % log_freq == 0:print(f'\rEpoch {epoch:4.0f}, cost: {cost:14.4f}, KL term: {kl:10.4f}, log likelihood part: {ll:10.4f}, accuracy: {accuracy:10.4f}, train time: {train_time:6.4f}, test time: {test_time:6.4f}')

    # predictions = np.mean(model.prediction(x_train, batch_size=batch_size), 0)
    # plt.scatter(y_train, predictions)
    # plt.plot([min(y_train), max(y_train)], [min(y_train), max(y_train)], color='r')
    # plt.show()

    return {'costs': costs, 'accuracies': accuracies}