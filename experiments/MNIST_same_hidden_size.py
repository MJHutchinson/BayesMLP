import pickle
from model.classification import BayesMLPClassification
from model.test_model import test_model
from data.data_loader import MnistDataloader

# batch_size = 1000
experiment_name = 'test/test1'
no_epochs = 100
hidden_layers = 1

hs = [10, 20, 30, 40, 50, 100, 200, 300, 400, 800, 1200]
bss = [1000, 1000, 1000, 1000, 1000, 500, 500, 200, 200, 100, 100]

# test MLP
data_gen = MnistDataloader()
input_size, train_length, ouput_size = data_gen.get_dims()

results = {}

for h, batch_size in zip(hs, bss):
    print(f'running model hidden size {h}')
    model = BayesMLPClassification(input_size, [h] * hidden_layers, ouput_size, train_length, no_pred_samples=10)

    result = test_model(model, data_gen, 20, batch_size)

    model.close_session()

    results[str(h)] = result

    with open(f'./results/{experiment_name}.pkl', 'wb') as h:
        pickle.dump(results, h, protocol=pickle.HIGHEST_PROTOCOL)