import os
import pickle
import yaml
from model.regression import BayesMLPRegression
from model.utils import test_model_regression
from utils.utils import num_to_name, get_search_space
from data.data_loader import get_loader_by_name

hidden_layers = 2
data_set = 'wine-quality-red'
log_dir = './results'
config_dir = './config'

h = [4, 4]
sizes = [0.0001, 0.001, 0.01, 0.1, 1,10,100,1000]

experiment_name = f'{hidden_layers}_layers'
results_file = f'{log_dir}/{data_set}/{experiment_name}_{h}_prior_sizes.pkl'
config_file = f'{config_dir}/{data_set}.yaml'

config = yaml.load(open(config_file, 'rb'))
epochs = config['epochs']
lr = config['learning_rate']

print(f'Running experiment on {data_set} with parameters:\n'
      f'{config}\n'
      f'Saving results in {results_file}\n')

data_loader = get_loader_by_name(data_set)
input_size, train_length, output_size = data_loader.get_dims()
_, _, y_mu, y_sigma = data_loader.get_transforms()

os.makedirs(f'./{log_dir}/{data_set}', exist_ok=True)

if os.path.isfile(results_file):
    results = pickle.load(open(results_file, 'rb'))
else:
    results = {}


for idx, size in enumerate(sizes):

    batch_size = 500

    if str(h) in results.keys():
        print(f'Already have {h}. Skipping')
        continue

    print(f'running model prior var {size}, div by parameter number, network {idx+1} of {len(sizes)}')
    model = BayesMLPRegression(input_size, h, output_size, train_length, y_mu, y_sigma, no_pred_samples=10, learning_rate=lr, prior_var=size, type=type)

    result = test_model_regression(model, data_loader, epochs, batch_size, log_freq=100)

    model.close_session()

    results[str(size)] = {'hidden_sizes': h, 'batch_size': batch_size, 'epochs': epochs, 'results': result}

    with open(results_file, 'wb') as hdl:
        pickle.dump(results, hdl)
