import cProfile
from data.data_loader import MnistDataloader
from model.classification import BayesMLPClassification
from model.utils import test_model

def test():
    data_gen = MnistDataloader()
    input_size, train_length, ouput_size = data_gen.get_dims()
    h = [1200, 1200]
    model = BayesMLPClassification(input_size, h, ouput_size, train_length, no_pred_samples=10)
    result = test_model(model, data_gen, 2, 500)
    model.close_session()


cProfile.run('test()')