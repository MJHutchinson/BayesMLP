import os
import pickle
from model.classification import BayesMLPClassification
from model.utils import test_model
from utils.utils import num_to_name
import itertools
from data.data_loader import RegressionDataloader
import matplotlib.pyplot as plt

hidden_layers = 2
data_set = 'mnist'
no_epochs = 20
log_dir = './results'

experiment_name = f'{data_set}/{num_to_name(hidden_layers)}_layers'
results_file = f'./results/{experiment_name}.pkl'

h = [2000, 2000]
batch_size = 100

# test MLP
data_gen = RegressionDataloader(data_set)
input_size, train_length, ouput_size = data_gen.get_dims()

print(f'running model hidden size {h}')
model = BayesMLPClassification(input_size, h, ouput_size, train_length, no_pred_samples=100)

result = test_model(model, data_gen, 1, batch_size)

model.close_session()

# x = [10, 50, 100, 500, 1000]
# x2 = [x**2 for x in x]
# x5 = [x**.5 for x in x]
# y = [7500, 7000, 3500, 900, 300]
#
# plt.plot(x, y)
# plt.show()
#
# plt.plot(x2, y)
# plt.show()
#
# plt.plot(x5, y)
# plt.show()
#
# plt.semilogx(x,y)
# plt.show()

