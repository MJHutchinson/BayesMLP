from model.regression import BayesMLPGen, BayesMLPRegression
import numpy as np
import matplotlib.pyplot as plt
from model.test_model import test_model_regression
import time

x = np.expand_dims(np.random.uniform(-1., 1., 1000), 1).astype(np.float32)

generator = BayesMLPGen(1, [25, 25, 25], 1, x.shape[0], 0., 1., no_pred_samples=1000)

preds = generator.prediction(x, 1000)
preds = np.mean(preds, axis=0)

generator.close_session()

finder = BayesMLPRegression(1, [25,25,25], 1, x.shape[0], 0., 1., prior_var=1.0, learning_rate=0.001)

epochs = 5000

kl = finder.sess.run(finder.KL)

for epoch in range(epochs):
    t = time.time()
    cost, kl, ll = finder.train_one(x, preds, batch_size=1000)
    train_time = time.time() - t
    logloss, rmse = finder.accuracy(x, preds, batch_size=1000)
    test_time = time.time() - train_time - t

    vy = finder.sess.run(finder.noise_var)

    if epoch % 10 == 0: print(
        f'\rEpoch {epoch:4.0f}, cost: {cost:14.4f}, KL term: {kl:10.4f}, train log likelihood term: {ll:10.4f}, test log likelihood: {logloss:10.4f}, test rmse: {rmse:10.4f}, log noise var: {vy:10f}, train time: {train_time:6.4f}, test time: {test_time:6.4f}')


preds2 = np.mean(finder.prediction(x, 100), axis=0)

plt.scatter(x, preds)
plt.scatter(x, preds2)
plt.show()
