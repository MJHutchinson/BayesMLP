import os
import pickle
import yaml
from model.classification import BayesMLPClassification
from model.regression import BayesMLPRegression
from model.utils import test_model
from utils.utils import num_to_name, get_search_space
from data.data_loader import get_loader_by_name

hidden_layers = 4
data_set = 'concrete'
log_dir = './results'
config_dir = './config'

experiment_name = f'{num_to_name(hidden_layers)}_layers'
results_file = f'{log_dir}/{data_set}/{experiment_name}.pkl'
config_file = f'{config_dir}/{data_set}.yaml'

config = yaml.load(open(config_file, 'rb'))
hs = config['hs']
epochs = config['epochs']
search_space = config['search_space']
lr = config['learning_rate']
hs = list(reversed(hs))

data_loader = get_loader_by_name(data_set)
input_size, train_length, output_size = data_loader.get_dims()

os.makedirs(f'./{log_dir}/{data_set}', exist_ok=True)

if os.path.isfile(results_file):
    results = pickle.load(open(results_file, 'rb'))
else:
    results = {}

search_space = get_search_space(search_space, hs, hidden_layers)

for idx, network in enumerate(search_space):
    h = [i for i in network]
    batch_size = data_loader.get_batch_size(max(h))

    if str(h) in results.keys():
        print(f'Already have {h}. Skipping')
        continue

    print(f'running model hidden size {h}, network {idx+1} of {len(search_space)}')
    model = BayesMLPRegression(input_size, h, output_size, train_length, no_pred_samples=10, learning_rate=lr)

    result = test_model(model, data_loader, epochs, batch_size)

    model.close_session()

    results[str(h)] = {'hidden_sizes': h, 'batch_size': batch_size, 'epochs': epochs, 'results': result}

    with open(results_file, 'wb') as h:
        pickle.dump(results, h)
