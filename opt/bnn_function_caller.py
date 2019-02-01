import os
import traceback
import model.run_model_tensorflow

from opt.nn_function_caller import NNFunctionCaller
from data.data_loader import RegressionDataloader
from time import sleep


_MAX_TRIES = 3
_SLEEP_BETWEEN_TRIES_SECS = 3

def get_default_params():
    """ default parametrs to run BNNs with """
    return {
        'batchSize': 1000,
        'epochs': 30000,
        'learningRate': 0.001
    }


class BNNMLPFunctionCaller(NNFunctionCaller):

    def __init__(self, *args, **kwargs):
        super(BNNMLPFunctionCaller, self).__init__(*args, **kwargs)

        self.data_loader = RegressionDataloader(self.train_params.data_set, self.train_params.data_dir)
        self.reporter.writeln('Loader data ' + self.train_params.data_set)
        self.reporter.writeln('Training data shape: ' + 'x: ' + str(self.data_loader.X_train.shape) + ', y: ' + str(self.data_loader.Y_train.shape))
        self.reporter.writeln('Training data shape: ' + 'x: ' + str(self.data_loader.X_test.shape) + ', y: ' + str(self.data_loader.Y_test.shape))

        if not hasattr(self.train_params, 'tf_params'):
            self.train_params.tf_params = get_default_params()

        self.train_params.tf_params['metric'] = self.train_params.metric

    def _eval_validation_score(self, point, qinfo):
        os.environ['CUDA_VISIBLE_DEVICES'] = str(qinfo.worker_id)
        num_tries = 0
        success = False

        while num_tries < _MAX_TRIES and not success:
            try:
                self.reporter.writeln(f'Running on gpu {qinfo.worker_id}: {qinfo}')
                if hasattr(qinfo, 'iteration'):
                    test_score = model.run_model_tensorflow.compute_validation_error(point,
                                                                                     self.data_loader,
                                                                                     self.train_params.tf_params,
                                                                                     qinfo.worker_id,
                                                                                     qinfo.log_dir,
                                                                                     name_prefix=f'{qinfo.iteration}')
                else:
                    test_score = model.run_model_tensorflow.compute_validation_error(point,
                                                                                     self.data_loader,
                                                                                     self.train_params.tf_params,
                                                                                     qinfo.worker_id,
                                                                                     qinfo.log_dir)
                success = True
            except Exception as e:
                sleep(_SLEEP_BETWEEN_TRIES_SECS)
                num_tries += 1
                self.reporter.writeln(f'********* Failed to try {num_tries} with gpu {qinfo.worker_id}')
                self.reporter.writeln(f'{e}')
                self.reporter.writeln(f'{traceback.format_exc()}')
                print(e)

        return test_score
