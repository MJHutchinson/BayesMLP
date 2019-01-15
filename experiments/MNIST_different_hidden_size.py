import pickle
from model.classification import BayesMLPClassification
from model.utils import test_model
from utils.file_utils import num_to_name
import itertools
from data.data_loader import MnistDataloader

hidden_layers = 2
data_set = 'mnist'
experiment_name = f'{data_set}/{num_to_name(hidden_layers)}_layers'
no_epochs = 20

hs = [10, 20, 30, 40, 50, 100, 200, 300, 400, 800, 1200]
bss = [1000, 1000, 1000, 1000, 1000, 500, 500, 200, 200, 100, 100]

# test MLP
data_gen = MnistDataloader()
input_size, train_length, ouput_size = data_gen.get_dims()

results = pickle.load(open(f'./results/{experiment_name}.pkl', 'rb'))

for combo in itertools.combinations(zip(reversed(hs), reversed(bss)), hidden_layers):
    h = [i[0] for i in combo]
    batch_size = min([i[1] for i in combo])

    if str(h) in results.keys():
        print(f'Already have {h}. Skipping')
        continue

    print(f'running model hidden size {h}')
    model = BayesMLPClassification(input_size, h, ouput_size, train_length, no_pred_samples=10)

    result = test_model(model, data_gen, no_epochs, batch_size)

    model.close_session()

    results[str(h)] = result

    with open(f'./results/{experiment_name}.pkl', 'wb') as h:
        pickle.dump(results, h, protocol=pickle.HIGHEST_PROTOCOL)
