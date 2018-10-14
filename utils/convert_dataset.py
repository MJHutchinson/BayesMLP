import numpy as np
import pickle
from utils import get_immediate_subdirectories

root_dir = './data_dir'
data_file = 'data.txt'
targets_file = 'index_target.txt'
features_file = 'index_features.txt'

split_test = 0.15
split_valid = 0.15

dirs = get_immediate_subdirectories('./data_dir')

for dir in dirs:
    data_path = f'{root_dir}/{dir}/data/{data_file}'
    targets_path = f'{root_dir}/{dir}/data/{targets_file}'
    features_path = f'{root_dir}/{dir}/data/{features_file}'

    data = np.loadtxt(data_path)
    index_targets = np.loadtxt(targets_path)
    index_features = np.loadtxt(features_path)

    np.random.seed(1)
    np.random.shuffle(data)

    x = data[:, [int(i) for i in index_features.tolist()]]
    y = data[:, int(index_targets.tolist())]

    test_start = int((1 - split_test - split_valid) * len(data))
    valid_start = int((1 - split_valid) * len(data))

    x_train = x[:test_start, :]
    y_train = y[:test_start]
    train_set = [x_train, y_train]

    x_test = x[test_start:valid_start, :]
    y_test = y[test_start:valid_start]
    test_set = [x_test, y_test]

    x_valid = x[valid_start:, :]
    y_valid = y[valid_start:]
    valid_set = [x_valid, y_valid]

    # with open(f'../data/{dir}.pkl', 'wb') as f:
    #     pickle.dump((train_set, test_set, valid_set), f)

    print(f'datset: {dir}, len:{len(data)}, train len: {len(x_train)}, test len: {len(x_test)}, valid len: {len(x_valid)}')

