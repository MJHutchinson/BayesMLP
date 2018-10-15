import time
import matplotlib.pyplot as plt
import numpy as np

def test_model(model, data_gen, epochs, batch_size=100):
    x_train, y_train, x_test, y_test = data_gen.get_data()

    costs = []
    accuracies = []

    for epoch in range(epochs):
        t = time.time()
        cost = model.train_one(x_train, y_train, batch_size=batch_size)
        train_time = time.time()-t
        accuracy = model.accuracy(x_test, y_test, batch_size=batch_size)
        test_time = time.time() - train_time - t

        costs.append(cost)
        accuracies.append(accuracy)
        print(f'\rEpoch {epoch}, cost: {cost}, accuracy: {accuracy}, train time: {train_time}, test time: {test_time}')

    predictions = np.mean(model.prediction(x_train, batch_size=batch_size), 0)
    plt.scatter(y_train, predictions)
    plt.plot([min(y_train), max(y_train)], [min(y_train), max(y_train)], color='r')
    plt.show()

    return {'costs': costs, 'accuracies': accuracies}