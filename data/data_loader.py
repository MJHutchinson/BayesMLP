import os
import pickle
import numpy as np


class Dataloader(object):
    def get_data(self):
        raise NotImplementedError

    def get_batch_size(self, max_hidden_layer_size):
        raise NotImplementedError

    @property
    def input_size(self):
        raise NotImplementedError

    @property
    def output_size(self):
        raise NotImplementedError

    @property
    def train_length(self):
        raise NotImplementedError

    @property
    def y_mu(self):
        raise NotImplementedError

    @property
    def y_sigma(self):
        raise NotImplementedError


class RegressionDataloader(Dataloader):
    
    def __init__(self, X_train, X_test, Y_train, Y_test):

        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test

        self.type = 'regression'

        if len(self.Y_train.shape) == 1:
            self.Y_train = np.expand_dims(self.Y_train, 1)
        if len(self.Y_test.shape) == 1:
            self.Y_test = np.expand_dims(self.Y_test, 1)

        self.X_means = 0.5 * (
                    np.expand_dims(np.max(self.X_train, axis=0), 0) + np.expand_dims(np.min(self.X_train, axis=0),
                                                                                     0))  # np.expand_dims(np.mean(self.X_train, axis=0), 0)
        self.Y_means = 0.5 * (
                    np.expand_dims(np.max(self.Y_train, axis=0), 0) + np.expand_dims(np.min(self.Y_train, axis=0),
                                                                                     0))  # np.expand_dims(np.mean(self.Y_train, axis=0), 0)
        self.X_sigmas = 0.5 * (
                    np.expand_dims(np.max(self.X_train, axis=0), 0) - np.expand_dims(np.min(self.X_train, axis=0),
                                                                                     0))  # np.sqrt(np.expand_dims(np.var(self.X_train, axis=0), 0)) #
        self.Y_sigmas = 0.5 * (
                    np.expand_dims(np.max(self.Y_train, axis=0), 0) - np.expand_dims(np.min(self.Y_train, axis=0),
                                                                                     0))  # np.sqrt(np.expand_dims(np.var(self.Y_train, axis=0), 0)) #

        self.X_train_transform, self.Y_train_transform = self.transform(self.X_train, self.Y_train)
        self.X_test_transform, self.Y_test_transform = self.transform(self.X_test, self.Y_test)

    def transform(self, X, Y):
        return (X - self.X_means) / self.X_sigmas, (Y - self.Y_means) / self.Y_sigmas

    ## Depreciated
    def get_dims(self):
        # Get data input and output dimensions
        return self.X_train.shape[1], self.X_train.shape[0], self.Y_train.shape[1]

    def get_data(self):
        return self.X_train_transform, self.Y_train_transform, self.X_test_transform, self.Y_test_transform
        # return self.X_train, self.Y_train, self.X_test, self.Y_test

    def get_transforms(self):
        return self.X_means, self.X_sigmas, self.Y_means, self.Y_sigmas

    def get_batch_size(self, max_hidden_layer_size):
        return NotImplementedError

    @property
    def input_size(self):
        return self.X_train.shape[1]

    @property
    def output_size(self):
        return self.Y_train.shape[1]

    @property
    def train_length(self):
        return self.X_train.shape[0]

    @property
    def y_mu(self):
        return self.Y_means

    @property
    def y_sigma(self):
        return self.Y_sigmas


class RegressionDataloaderFixedSplits(RegressionDataloader):
    def __init__(self, pickle_name, data_dir='./data_dir'):
        self.pickle_name = pickle_name

        f = open(f'{data_dir}/{self.pickle_name}.pkl', 'rb')
        train_set, valid_set, test_set = pickle.load(f)
        f.close()

        X_train = np.vstack((train_set[0], valid_set[0]))
        Y_train = np.hstack((train_set[1], valid_set[1]))
        X_test = test_set[0]
        Y_test = test_set[1]
        
        super().__init__(X_train, X_test, Y_train, Y_test)


class RegressionDataloaderVariableSplits(RegressionDataloader):

    def __init__(self, data_path, data_set, split_number=0):
        
        data = np.loadtxt(os.path.join(data_path, data_set, 'data.txt'))
        feature_index = np.loadtxt(os.path.join(data_path, data_set, 'index_features.txt'))
        target_index = np.loadtxt(os.path.join(data_path, data_set, 'index_target.txt'))

        X = data[:, [int(i) for i in feature_index.tolist()]]
        Y = data[:, int(target_index.tolist())]
        
        train_index = np.loadtxt(os.path.join(data_path, data_set, f'index_train_{split_number}.txt'))
        test_index = np.loadtxt(os.path.join(data_path, data_set, f'index_test_{split_number}.txt'))

        X_train = X[[int(i) for i in train_index.tolist()]]
        Y_train = Y[[int(i) for i in train_index.tolist()]]

        X_test = X[[int(i) for i in test_index.tolist()]]
        Y_test = Y[[int(i) for i in test_index.tolist()]]

        super().__init__(X_train, X_test, Y_train, Y_test)


class ClassificationDataloader(Dataloader):
    def __init__(self, pickle_name, data_dir='./data_dir', batch_size=100):
        self.pickle_name = pickle_name

        self.type = 'classification'
        self.batch_size = batch_size

        f = open(f'{data_dir}/{self.pickle_name}.pkl', 'rb')
        train_set, valid_set, test_set = pickle.load(f)
        f.close()

        self.X_train = np.vstack((train_set[0], valid_set[0]))
        self.Y_train = np.hstack((train_set[1], valid_set[1])).astype(np.int)

        self.classes = int(np.max(self.Y_train) + 1)
        self.Y_train = np.eye(self.classes)[self.Y_train]

        self.X_test = test_set[0]
        self.Y_test = test_set[1].astype(np.int)
        self.Y_test = np.eye(self.classes)[self.Y_test]

        # if len(self.Y_train.shape) == 1:
        #     self.Y_train = np.expand_dims(self.Y_train, 1)
        # if len(self.Y_test.shape) == 1:
        #     self.Y_test = np.expand_dims(self.Y_test, 1)

        ## Batching things

        self.N = self.X_train.shape[0]
        if batch_size > self.N:
            batch_size = self.N

        perm_inds = list(range(self.X_train.shape[0]))
        np.random.shuffle(perm_inds)
        self.cur_x_train = self.X_train[perm_inds]
        self.cur_y_train = self.Y_train[perm_inds]

        self.total_batches = int(np.ceil(self.N * 1.0 / batch_size))
        self.curr_batch = 0


    def get_dims(self):
        # Get data input and output dimensions
        return self.X_train.shape[1], self.X_train.shape[0], self.classes

    def get_data(self):
        return self.X_train, self.Y_train, self.X_test, self.Y_test

    def get_batch_size(self, max_hidden_layer_size):
        return NotImplementedError

    @property
    def input_size(self):
        return self.X_train.shape[1]

    @property
    def output_size(self):
        return self.Y_train.shape[1]

    @property
    def train_length(self):
        return self.X_train.shape[0]

    @property
    def y_mu(self):
        return 0

    @property
    def y_sigma(self):
        return 1

    def next_train_batch(self):
        if self.curr_batch >= self.total_batches:
            self.curr_batch = 0
            perm_inds = list(range(self.X_train.shape[0]))
            np.random.shuffle(perm_inds)
            self.cur_x_train = self.X_train[perm_inds]
            self.cur_y_train = self.Y_train[perm_inds]

        start_ind = self.curr_batch * self.batch_size
        end_ind = np.min([(self.curr_batch + 1) * self.batch_size, self.N])

        batch_x = self.cur_x_train[start_ind:end_ind, :]
        batch_y = self.cur_y_train[start_ind:end_ind, :]

        self.curr_batch += 1

        return batch_x, batch_y


# class DummyDataloader(RegressionDataloader):
#     def __init__(self):
#         self.pickle_name = 'dummy'
#
#         a = 20
#         b = 1
#         c = 3
#
#         len = 10000
#
#         noise = 0.01
#
#         self.X_train = np.expand_dims(np.random.uniform(0, 10, len), 1)
#         self.Y_train = (self.X_train ** 2 * a) + (self.X_train * b) + c + np.expand_dims(
#             np.random.normal(0, noise, len), 1)
#         self.X_test = np.expand_dims(np.random.uniform(0, 10, len), 1)
#         self.Y_test = (self.X_test ** 2 * a) + (self.X_test * b) + c + np.expand_dims(np.random.normal(0, noise, len),
#                                                                                       1)
#
#         self.X_means = 0.5 * (
#                     np.expand_dims(np.max(self.X_train, axis=0), 0) + np.expand_dims(np.min(self.X_train, axis=0),
#                                                                                      0))  # np.expand_dims(np.mean(self.X_train, axis=0), 0)
#         self.Y_means = 0.5 * (
#                     np.expand_dims(np.max(self.Y_train, axis=0), 0) + np.expand_dims(np.min(self.Y_train, axis=0),
#                                                                                      0))  # np.expand_dims(np.mean(self.Y_train, axis=0), 0)
#         self.X_sigmas = 0.5 * (
#                     np.expand_dims(np.max(self.X_train, axis=0), 0) - np.expand_dims(np.min(self.X_train, axis=0),
#                                                                                      0))  # np.sqrt(np.expand_dims(np.var(self.X_train, axis=0), 0)) #
#         self.Y_sigmas = 0.5 * (
#                     np.expand_dims(np.max(self.Y_train, axis=0), 0) - np.expand_dims(np.min(self.Y_train, axis=0),
#                                                                                      0))  # np.sqrt(np.expand_dims(np.var(self.Y_train, axis=0), 0)) #
#
#         self.X_train_transform, self.Y_train_transform = self.transform(self.X_train, self.Y_train)
#         self.X_test_transform, self.Y_test_transform = self.transform(self.X_test, self.Y_test)
#
#         self.classes = self.Y_train.shape[1]
#
#     def get_batch_size(self, max_hidden_layer_size):
#         return 1000
#
#
# class BostonHousingDataloader(RegressionDataloader):
#     def __init__(self):
#         super().__init__('bostonHousing')
#
#     def get_batch_size(self, max_hidden_layer_size):
#         return 500
#
#
# class ConcreteDataloader(RegressionDataloader):
#     def __init__(self):
#         super().__init__('concrete')
#
#     def get_batch_size(self, max_hidden_layer_size):
#         return 1000
#
#
# class EnergyDataloader(RegressionDataloader):
#     def __init__(self):
#         super().__init__('energy')
#
#
# class Kin8nmDataloader(RegressionDataloader):
#     def __init__(self):
#         super().__init__('kin8nm')
#
#     def get_batch_size(self, max_hidden_layer_size):
#         return 1000
#
#
# ''  # '#'#
#
#
# class MnistDataloader(ClassificationDataloader):
#     def __init__(self):
#         super().__init__('mnist')
#
#     def get_batch_size(self, max_hidden_layer_size):
#         if max_hidden_layer_size <= 50:
#             return 7000
#         elif max_hidden_layer_size <= 100:
#             return 3500
#         elif max_hidden_layer_size <= 500:
#             return 900
#         elif max_hidden_layer_size <= 1000:
#             return 300
#         else:
#             return 50
#
#
# class NavalPropulsionPlantDataloader(RegressionDataloader):
#     def __init__(self):
#         super().__init__('naval-propulsion-plant')
#
#     def get_batch_size(self, max_hidden_layer_size):
#         return 1000
#
#
# class PowerPlantDataloader(RegressionDataloader):
#     def __init__(self):
#         super().__init__('power-plant')
#
#     def get_batch_size(self, max_hidden_layer_size):
#         return 2000
#
#
# class ProteinTertiaryStructureDataloader(RegressionDataloader):
#     def __init__(self):
#         super().__init__('protein-tertiary-structure')
#
#     def get_batch_size(self, max_hidden_layer_size):
#         return 2000
#
#
# class WineQualityRedDataloader(RegressionDataloader):
#     def __init__(self):
#         super().__init__('wine-quality-red')
#
#     def get_batch_size(self, max_hidden_layer_size):
#         return 1000
#
#
# class YachtDataloader(RegressionDataloader):
#     def __init__(self):
#         super().__init__('yacht')
#
#     def get_batch_size(self, max_hidden_layer_size):
#         return 2000
#
#
# def get_all_dataloaders():
#     return [
#         BostonHousingDataloader(),
#         ConcreteDataloader(),
#         EnergyDataloader(),
#         Kin8nmDataloader(),
#         MnistDataloader(),
#         NavalPropulsionPlantDataloader(),
#         PowerPlantDataloader(),
#         ProteinTertiaryStructureDataloader(),
#         WineQualityRedDataloader(),
#         YachtDataloader(),
#         DummyDataloader()
#     ]
#
#
# def get_loader_by_name(name):
#     for dl in get_all_dataloaders():
#         if dl.pickle_name == name:
#             return dl
#
#     return None
