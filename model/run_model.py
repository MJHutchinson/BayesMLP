import os
from opt.nn_function_caller import NNFunctionCaller
from data.data_loader import RegressionDataloader

_MAX_TRIES = 3
_SLEEP_BETWEEN_TRIES_SECS = 3

def get_default_params():
    """ default parametrs to run BNNs with """
    return {
        'batchSize': 1000,
        'epochs': 30000,
        'learningRate': 0.001
    }


def build_model(nn):
    pass


class BNNMLPFunctionCaller(NNFunctionCaller):

    def __init__(self, *args, **kwargs):
        super(BNNMLPFunctionCaller, self).__init__(*args, **kwargs)

        self.data_loader = RegressionDataloader(self.train_params.data_set, self.train_params.data_dir)
        self.reporter.writeln('Loader data ' + self.train_params.data_set)
        self.reporter.writeln('Training data shape: ' + 'x: ' + self.data_loader.X_train.shape + ', y: ' + self.data_loader.Y_train.shape)
        self.reporter.writeln('Training data shape: ' + 'x: ' + self.data_loader.X_test.shape + ', y: ' + self.data_loader.Y_test.shape)
        if not hasattr(self.train_params, 'tf_params'):
            self.train_params = get_default_params()

    def _eval_validation_score(self, qinfo, nn):
        os.environ['CUDA_VISIBLE_DEVICES'] = str(qinfo.worker_id)
        num_tries = 0

        while num_tries < _MAX_TRIES:
            valid_score = 