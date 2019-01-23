import pickle
import numpy as np


class RegressionDataloader():
    def __init__(self, pickle_name, data_dir='./data_dir'):
        self.pickle_name = pickle_name

        f = open(f'{data_dir}/{self.pickle_name}.pkl', 'rb')
        train_set, valid_set, test_set = pickle.load(f)
        f.close()

        self.X_train = np.vstack((train_set[0], valid_set[0]))
        self.Y_train = np.hstack((train_set[1], valid_set[1]))
        self.X_test = test_set[0]
        self.Y_test = test_set[1]

        if len(self.Y_train.shape) == 1:
            self.Y_train = np.expand_dims(self.Y_train, 1)
        if len(self.Y_test.shape) == 1:
            self.Y_test = np.expand_dims(self.Y_test, 1)

        self.X_means = 0.5 * (np.expand_dims(np.max(self.X_train, axis=0), 0) + np.expand_dims(np.min(self.X_train, axis=0), 0)) # np.expand_dims(np.mean(self.X_train, axis=0), 0)
        self.Y_means = 0.5 * (np.expand_dims(np.max(self.Y_train, axis=0), 0) + np.expand_dims(np.min(self.Y_train, axis=0), 0)) # np.expand_dims(np.mean(self.Y_train, axis=0), 0)
        self.X_sigmas = 0.5 * (np.expand_dims(np.max(self.X_train, axis=0), 0) - np.expand_dims(np.min(self.X_train, axis=0), 0)) # np.sqrt(np.expand_dims(np.var(self.X_train, axis=0), 0)) #
        self.Y_sigmas = 0.5 * (np.expand_dims(np.max(self.Y_train, axis=0), 0) - np.expand_dims(np.min(self.Y_train, axis=0), 0)) # np.sqrt(np.expand_dims(np.var(self.Y_train, axis=0), 0)) #

        self.X_train_transform, self.Y_train_transform = self.transform(self.X_train, self.Y_train)
        self.X_test_transform, self.Y_test_transform = self.transform(self.X_test, self.Y_test)

    def transform(self, X, Y):
        return (X-self.X_means)/self.X_sigmas, (Y - self.Y_means)/self.Y_sigmas

    def antitransform(self, X, Y):
        return (X*self.X_sigmas) + self.X_means, (Y*self.Y_sigmas) + self.Y_means

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


class ClassificationDataloader():
    def __init__(self, pickle_name, data_dir='./data_dir'):
        self.pickle_name = pickle_name

        f = open(f'{data_dir}/{self.pickle_name}.pkl', 'rb')
        train_set, valid_set, test_set = pickle.load(f)
        f.close()

        self.X_train = np.vstack((train_set[0], valid_set[0]))
        self.Y_train = np.hstack((train_set[1], valid_set[1])).astype(np.int)
        self.X_test = test_set[0]
        self.Y_test = test_set[1].astype(np.int)
        self.classes = int(np.max(self.Y_train)+1)

    def get_dims(self):
        # Get data input and output dimensions
        return self.X_train.shape[1], self.X_train.shape[0], self.classes

    def get_data(self):
        return self.X_train, np.eye(self.classes)[self.Y_train], self.X_test, np.eye(self.classes)[self.Y_test]

    def get_batch_size(self, max_hidden_layer_size):
        return NotImplementedError


class DummyDataloader(RegressionDataloader):
    def __init__(self):
        self.pickle_name = 'dummy'

        a = 20
        b = 1
        c = 3

        len = 10000

        noise = 0.01

        self.X_train = np.expand_dims(np.random.uniform(0,10, len), 1)
        self.Y_train = (self.X_train ** 2 * a) + (self.X_train * b) + c + np.expand_dims(np.random.normal(0, noise, len), 1)
        self.X_test = np.expand_dims(np.random.uniform(0,10, len), 1)
        self.Y_test = (self.X_test ** 2 * a) + (self.X_test * b) + c + np.expand_dims(np.random.normal(0, noise, len), 1)

        self.X_means = 0.5 * (np.expand_dims(np.max(self.X_train, axis=0), 0) + np.expand_dims(np.min(self.X_train, axis=0), 0))  # np.expand_dims(np.mean(self.X_train, axis=0), 0)
        self.Y_means = 0.5 * (np.expand_dims(np.max(self.Y_train, axis=0), 0) + np.expand_dims(np.min(self.Y_train, axis=0), 0))  # np.expand_dims(np.mean(self.Y_train, axis=0), 0)
        self.X_sigmas = 0.5 * (np.expand_dims(np.max(self.X_train, axis=0), 0) - np.expand_dims(np.min(self.X_train, axis=0), 0))  # np.sqrt(np.expand_dims(np.var(self.X_train, axis=0), 0)) #
        self.Y_sigmas = 0.5 * (np.expand_dims(np.max(self.Y_train, axis=0), 0) - np.expand_dims(np.min(self.Y_train, axis=0), 0))  # np.sqrt(np.expand_dims(np.var(self.Y_train, axis=0), 0)) #

        self.X_train_transform, self.Y_train_transform = self.transform(self.X_train, self.Y_train)
        self.X_test_transform, self.Y_test_transform = self.transform(self.X_test, self.Y_test)

        self.classes = self.Y_train.shape[1]

    def get_batch_size(self, max_hidden_layer_size):
        return 1000


class BostonHousingDataloader(RegressionDataloader):
    def __init__(self):
        super().__init__('bostonHousing')

    def get_batch_size(self, max_hidden_layer_size):
        return 500


class ConcreteDataloader(RegressionDataloader):
    def __init__(self):
        super().__init__('concrete')

    def get_batch_size(self, max_hidden_layer_size):
        return 1000


class EnergyDataloader(RegressionDataloader):
    def __init__(self):
        super().__init__('energy')


class Kin8nmDataloader(RegressionDataloader):
    def __init__(self):
        super().__init__('kin8nm')

    def get_batch_size(self, max_hidden_layer_size):
        return 1000

''#'#'#
class MnistDataloader(ClassificationDataloader):
    def __init__(self):
        super().__init__('mnist')

    def get_batch_size(self, max_hidden_layer_size):
        if max_hidden_layer_size <= 50:
            return  7000
        elif max_hidden_layer_size <= 100:
            return 3500
        elif max_hidden_layer_size <= 500:
            return 900
        elif max_hidden_layer_size <= 1000:
            return 300
        else:
            return 50


class NavalPropulsionPlantDataloader(RegressionDataloader):
    def __init__(self):
        super().__init__('naval-propulsion-plant')

    def get_batch_size(self, max_hidden_layer_size):
        return 1000


class PowerPlantDataloader(RegressionDataloader):
    def __init__(self):
        super().__init__('power-plant')

    def get_batch_size(self, max_hidden_layer_size):
        return 2000


class ProteinTertiaryStructureDataloader(RegressionDataloader):
    def __init__(self):
        super().__init__('protein-tertiary-structure')
        
    def get_batch_size(self, max_hidden_layer_size):
        return 2000


class WineQualityRedDataloader(RegressionDataloader):
    def __init__(self):
        super().__init__('wine-quality-red')
        
    def get_batch_size(self, max_hidden_layer_size):
        return 1000


class YachtDataloader(RegressionDataloader):
    def __init__(self):
        super().__init__('yacht')
        
    def get_batch_size(self, max_hidden_layer_size):
        return 2000


def get_all_dataloaders():
    return [
        BostonHousingDataloader(),
        ConcreteDataloader(),
        EnergyDataloader(),
        Kin8nmDataloader(),
        MnistDataloader(),
        NavalPropulsionPlantDataloader(),
        PowerPlantDataloader(),
        ProteinTertiaryStructureDataloader(),
        WineQualityRedDataloader(),
        YachtDataloader(),
        DummyDataloader()
    ]


def get_loader_by_name(name):
    for dl in get_all_dataloaders():
        if dl.pickle_name == name:
            return dl

    return None
