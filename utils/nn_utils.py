from scipy.sparse import dok_matrix
from model.neural_network_representation import get_mlp_layer_labels, MultiLayerPerceptron

def get_feedforward_adj_mat(num_layers):
    """ Returns an adjacency matrix for a feed forward network. """
    ret = dok_matrix((num_layers, num_layers))
    for i in range(num_layers - 1):
        ret[i, i + 1] = 1
    return ret


def get_feedforward_nn(hidden_size, hidden_layer):
    hidden_configuration = [hidden_size] * hidden_layer
    all_layer_labels = get_mlp_layer_labels('reg')
    layer_labels = ['ip'] + ['relu'] * len(hidden_configuration) + ['linear', 'op']
    num_units_each_layer = [None] + hidden_configuration + [None, None]
    A = get_feedforward_adj_mat(len(layer_labels))

    return MultiLayerPerceptron('reg', layer_labels, A, num_units_each_layer, all_layer_labels)
