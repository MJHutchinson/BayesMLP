from scipy.sparse import dok_matrix


def get_feedforward_adj_mat(num_layers):
    """ Returns an adjacency matrix for a feed forward network. """
    ret = dok_matrix((num_layers, num_layers))
    for i in range(num_layers - 1):
        ret[i, i + 1] = 1
    return ret