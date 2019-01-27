###### WARNING: Likely Broken ######


import os
import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow_probability.python.distributions as distributions
import numpy as np
from copy import deepcopy

np.random.seed(0)
tf.set_random_seed(0)

os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
tf.logging.set_verbosity(tf.logging.WARN)

INITIAL_LOG_NOISE = -6.


""" Bayes MLP model using TFP"""


class BaysMLPRegressionTFP(Reg_NN):
    def __init__(self, input_size, hidden_size, output_size, training_size, y_mu, y_sigma,
                 no_train_samples=10, no_pred_samples=100, prev_means=None, prev_log_variances=None,
                 learning_rate=0.001,
                 prior_mean=0., prior_var=1.):

        super(BaysMLPRegressionTFP, self).__init__(input_size, hidden_size, output_size, training_size, y_mu, y_sigma)

        self.prior_mean = prior_mean
        self.prior_var = prior_var

        self.output_log_variance = tf.Variable(initial_value=INITIAL_LOG_NOISE, name='log_noise_variance')
        self.output_sigma = tf.exp(self.output_log_variance)

        self.no_train_samples = no_train_samples
        self.no_pred_samples = no_pred_samples
        self.training_size = training_size

        self._build_model(hidden_size, output_size)
        self._build_functions()

        self.assign_optimizer(learning_rate)
        self.assign_session()
        self.make_metrics()

    def _prediction(self, inputs, no_samples):
        with tf.name_scope('Sample_multiply'):
            K = no_samples
            input_shape = inputs.get_shape().as_list()
            act = tf.tile(tf.expand_dims(inputs, 0), [K, 1, 1])
            act = tf.reshape(act, [-1] + input_shape[1:])
            input_shape = tf.shape(inputs)

        output = self.model(act)
        output_shape = tf.shape(output)

        with tf.name_scope('Sample_unmultiply'):
            output = tf.reshape(output, tf.stack([tf.constant([-1])[0], input_shape[0], output_shape[-1]], 0))

        return output


    def _build_model(self, hidden_size, output_size):
        layers = []

        for idx, hs in enumerate(hidden_size):
            layers.append(tfp.layers.DenseLocalReparameterization(hs, activation=tf.nn.relu, name=f'layer_{idx}'))

        layers.append(tfp.layers.DenseLocalReparameterization(output_size, activation=None, name='output_layer'))

        self.model = tf.keras.Sequential(layers, 'Model')
        out = self.model(self.x)

    def _build_functions(self):
        # def train predictions and training metric production
        self.pred_train = self._prediction(self.x, self.no_train_samples)

        self.pred_train_actual = self.pred_train * self.y_sigma + self.y_mu
        self.targ_train_actual = self.y * self.y_sigma + self.y_mu

        self.loglik = _loglik(self.pred_train, self.y, self.output_sigma)
        self.KL = tf.div(sum(self.model.losses), self.training_size)

        self.cost = -self.loglik + self.KL

        # def test predictions and testing metrics
        self.pred_test = self._prediction(self.x, self.no_pred_samples)

        self.pred_test_actual = self.pred_test * self.y_sigma + self.y_mu
        self.targ_test_actual = self.y * self.y_sigma + self.y_mu

        self.mse = _mse(self.pred_test_actual, self.targ_test_actual)
        self.test_loglik = _test_loglik(self.pred_test, self.y, self.output_sigma)


""" Bayesian MLP Model """


""" Bayesian MLP Model with optional skips """


class BayesSkipMLPRegression(Reg_NN):
    def __init__(self, input_size, hidden_size, skips, output_size, training_size, y_mu, y_sigma,
                 no_train_samples=10, no_pred_samples=100, prev_means=None, prev_log_variances=None,
                 learning_rate=0.001,
                 prior_mean=0., prior_var=1.):

        super(BayesSkipMLPRegression, self).__init__(input_size, hidden_size, output_size, training_size, y_mu, y_sigma)

        self.config = {
            'hidden_size': hidden_size,
            'skip_connects': skips,
            'learning_rate': learning_rate,
            'prior_var': prior_var
        }

        m, v, self.size = self.create_weights(input_size, hidden_size, skips, output_size, prev_means, prev_log_variances)
        self.W_m, self.b_m = m[0], m[1]
        self.W_v, self.b_v = v[0], v[1]
        self.weights = [m, v]

        m, v = self.create_prior(input_size, hidden_size, skips, output_size, prev_means, prev_log_variances, prior_mean, prior_var)
        self.prior_W_m, self.prior_b_m = m[0], m[1]
        self.prior_W_v, self.prior_b_v = v[0], v[1]
        self.priors = [m, v]


        self.output_log_variance = tf.Variable(initial_value=INITIAL_LOG_NOISE, name='log_noise_variance')
        self.output_sigma = tf.exp(0.5 * self.output_log_variance)

        self.no_layers = len(self.size)-1
        self.no_train_samples = no_train_samples
        self.no_pred_samples = no_pred_samples
        self.training_size = training_size
        self.skips = skips


        # def train predictions and training metric production
        self.pred_train = self._prediction(self.x, no_train_samples)

        self.pred_train_actual = self.pred_train * self.y_sigma + self.y_mu
        self.targ_train_actual = self.y * self.y_sigma + self.y_mu

        self.loglik = _loglik(self.pred_train, self.y, self.output_sigma)
        self.KL = tf.div(_KL_term(self.weights, self.priors), self.training_size)

        self.cost = -self.loglik + self.KL


        # def test predictions and testing metrics
        self.pred_test  = self._prediction(self.x, self.no_pred_samples)

        self.pred_test_actual = self.pred_test * self.y_sigma + self.y_mu
        self.targ_test_actual = self.y * self.y_sigma + self.y_mu

        self.mse = _mse(self.pred_test_actual, self.targ_test_actual)
        self.test_loglik = _test_loglik(self.pred_test, self.y, self.output_sigma)


        self.assign_optimizer(learning_rate)
        self.assign_session()
        self.make_metrics()

    def _prediction(self, inputs, no_samples):
        return self._prediction_layer(inputs, no_samples)

        # this samples a layer at a time
    def _prediction_layer(self, inputs, no_samples):
        with tf.name_scope('Expane_sample'):
            K = no_samples
            N = tf.shape(inputs)[0]
            act = tf.tile(tf.expand_dims(inputs, 0), [K, 1, 1])
        act_prev = [act]

        with tf.name_scope('model/'):
            for i in range(self.no_layers - 1):
                with tf.name_scope(f'layer_{i}/'):
                    din = self.size[i]
                    dout = self.size[i + 1]

                    act_in = [act]

                    for skip in self.skips:
                        if skip[1] == i:
                            din += self.size[skip[0]]
                            act_in.append(act_prev[skip[0]])

                    act = tf.concat(act_in, -1)

                    m_pre = tf.einsum('kni,io->kno', act, self.W_m[i])
                    v_pre = tf.einsum('kni,io->kno', act ** 2.0, tf.exp(self.W_v[i]))
                    eps_w = tf.random_normal([K, N, dout], 0.0, 1.0, dtype=tf.float32)
                    pre_W = eps_w * tf.sqrt(1e-9 + v_pre) + m_pre
                    eps_b = tf.random_normal([K, 1, dout], 0.0, 1.0, dtype=tf.float32)
                    pre_b = eps_b * tf.exp(0.5 * self.b_v[i]) + self.b_m[i]
                    pre = pre_W + pre_b
                    act = tf.nn.relu(pre)
                    act_prev.append(act)

            with tf.name_scope(f'layer_{self.no_layers-1}/'):
                din = self.size[-2]
                dout = self.size[-1]

                act_in = [act]

                for skip in self.skips:
                    if skip[1] == self.no_layers-1:
                        din += self.size[skip[0]]
                        act_in.append(act_prev[skip[0]])

                act = tf.concat(act_in, -1)

                m_pre = tf.einsum('kni,io->kno', act, self.W_m[-1])
                v_pre = tf.einsum('kni,io->kno', act ** 2.0, tf.exp(self.W_v[-1]))
                eps_w = tf.random_normal([K, N, dout], 0.0, 1.0, dtype=tf.float32)
                pre_W = eps_w * tf.sqrt(1e-9 + v_pre) + m_pre
                eps_b = tf.random_normal([K, 1, dout], 0.0, 1.0, dtype=tf.float32)
                pre_b = eps_b * tf.exp(0.5 * self.b_v[-1]) + self.b_m[-1]
                pre = pre_W + pre_b

                return pre


    def _KL_term(self):
        with tf.name_scope('kl'):
            kl = 0
            for i in range(self.no_layers):
                with tf.name_scope(f'layer_{i}'):

                    m, v = self.W_m[i], self.W_v[i]
                    m0, v0 = self.prior_W_m[i], self.prior_W_v[i]

                    prior = distributions.Normal(m0, tf.sqrt(v0))
                    weights = distributions.Normal(m, tf.exp(0.5 * v))
                    kl += tf.reduce_sum(distributions.kl_divergence(weights, prior))

                    # const_term = -0.5 * dout * din
                    # log_std_diff = 0.5 * tf.reduce_sum(np.log(v0) - v)
                    # mu_diff_term = 0.5 * tf.reduce_sum((tf.exp(v) + (m0 - m) ** 2) / v0)
                    # kl += const_term + log_std_diff + mu_diff_term

                    m, v = self.b_m[i], self.b_v[i]
                    m0, v0 = self.prior_b_m[i], self.prior_b_v[i]

                    prior = distributions.Normal(m0, tf.sqrt(v0))
                    weights = distributions.Normal(m, tf.exp(0.5 * v))
                    kl += tf.reduce_sum(distributions.kl_divergence(weights, prior))

                    # const_term = -0.5 * dout
                    # log_std_diff = 0.5 * tf.reduce_sum(np.log(v0) - v)
                    # mu_diff_term = 0.5 * tf.reduce_sum((tf.exp(v) + (m0 - m) ** 2) / v0)
                    # kl += const_term + log_std_diff + mu_diff_term

            return kl

    def create_weights(self, in_dim, hidden_size, skips, out_dim, prev_weights, prev_variances):
        hidden_size = deepcopy(hidden_size)
        hidden_size.append(out_dim)
        hidden_size.insert(0, in_dim)
        no_params = 0
        no_layers = len(hidden_size) - 1
        W_m = []
        b_m = []
        W_v = []
        b_v = []
        with tf.name_scope('model/'):
            for i in range(no_layers):
                with tf.name_scope(f'layer_{i}/'):
                    din = hidden_size[i]
                    dout = hidden_size[i + 1]

                    for skip in skips:
                        if skip[1] == i:
                            din += hidden_size[skip[0]]

                    if prev_weights is None:
                        Wi_m_val = tf.truncated_normal([din, dout], stddev=0.1)
                        bi_m_val = tf.truncated_normal([dout], stddev=0.1)
                        Wi_v_val = tf.constant(-11.0, shape=[din, dout])
                        bi_v_val = tf.constant(-11.0, shape=[dout])
                    else:
                        Wi_m_val = prev_weights[0][i]
                        bi_m_val = prev_weights[1][i]
                        if prev_variances is None:
                            Wi_v_val = tf.constant(-11.0, shape=[din, dout])
                            bi_v_val = tf.constant(-11.0, shape=[dout])
                        else:
                            Wi_v_val = prev_variances[0][i]
                            bi_v_val = prev_variances[1][i]

                    Wi_m = tf.Variable(Wi_m_val, name='W_mean')
                    bi_m = tf.Variable(bi_m_val, name='b_mean')
                    Wi_v = tf.Variable(Wi_v_val, name='W_var')
                    bi_v = tf.Variable(bi_v_val, name='b_var')
                    W_m.append(Wi_m)
                    b_m.append(bi_m)
                    W_v.append(Wi_v)
                    b_v.append(bi_v)

                    tf.summary.histogram("weights_mean", Wi_m)
                    tf.summary.histogram("bias_mean", bi_m)
                    tf.summary.histogram("weights_variance", Wi_v)
                    tf.summary.histogram("bais_variance", bi_v)

            return [W_m, b_m], [W_v, b_v], hidden_size


    def create_prior(self, in_dim, hidden_size, skips, out_dim, prev_weights, prev_variances, prior_mean, prior_var):
        hidden_size = deepcopy(hidden_size)
        hidden_size.append(out_dim)
        hidden_size.insert(0, in_dim)
        no_params = 0
        no_layers = len(hidden_size) - 1
        W_m = []
        b_m = []
        W_v = []
        b_v = []
        for i in range(no_layers):
            din = hidden_size[i]
            dout = hidden_size[i + 1]

            for skip in skips:
                if skip[1] == i:
                    din += hidden_size[skip[0]]

            if prev_weights is not None and prev_variances is not None:
                Wi_m = prev_weights[0][i]
                bi_m = prev_weights[1][i]
                Wi_v = np.exp(prev_variances[0][i])
                bi_v = np.exp(prev_variances[1][i])
            else:
                Wi_m = prior_mean
                bi_m = prior_mean
                Wi_v = prior_var
                bi_v = prior_var

            W_m.append(Wi_m)
            b_m.append(bi_m)
            W_v.append(Wi_v)
            b_v.append(bi_v)

        return [W_m, b_m], [W_v, b_v]

""" Bayes MLP for regression built from NN representation from OTTOMAN paper """

class BayesMLPNNRegression(Reg_NN):
    def __init__(self, input_size, nn, training_size, y_mu, y_sigma,
                 no_train_samples=10, no_pred_samples=100, prev_means=None, prev_log_variances=None,
                 learning_rate=0.001,
                 prior_mean=0., prior_var=1.):

        super(BayesMLPNNRegression, self).__init__(input_size, 0, 1, training_size, y_mu, y_sigma)

        self.config = {
            'hidden_sizes': nn.num_units_in_each_layer,
            'con_mat': nn.conn_mat,
            'learning_rate': learning_rate,
            'prior_var': prior_var
        }


        m_w, v_w, m_p, v_p = self.create_weights_prior(self.x, nn, prior_mean, prior_var)

        self.W_m, self.b_m = m_w[0], m_w[1]
        self.W_v, self.b_v = v_w[0], v_w[1]

        self.prior_W_m, self.prior_b_m = m_p[0], m_p[1]
        self.prior_W_v, self.prior_b_v = v_p[0], v_p[1]

        self.weights = [m_w, v_w]
        self.priors =  [m_p, v_p]

        self.no_train_samples = no_train_samples
        self.no_pred_samples = no_pred_samples
        self.training_size = training_size

        self.output_log_variance = tf.Variable(initial_value=INITIAL_LOG_NOISE, name='log_noise_variance')
        self.output_sigma = tf.exp(0.5 * self.output_log_variance)


        # def train predictions and training metric production
        self.pred_train = self._prediction(self.x, nn, self.no_train_samples)

        self.pred_train_actual = self.pred_train * self.y_sigma + self.y_mu
        self.targ_train_actual = self.y * self.y_sigma + self.y_mu

        self.loglik = _loglik(self.pred_train, self.y, self.output_sigma)
        self.KL = tf.div(_KL_term(self.weights, self.priors), self.training_size)

        self.cost = -self.loglik + self.KL


        # def test predictions and testing metrics
        self.pred_test = self._prediction(self.x, nn, self.no_pred_samples)

        self.pred_test_actual = self.pred_test * self.y_sigma + self.y_mu
        self.targ_test_actual = self.y * self.y_sigma + self.y_mu

        self.mse = _mse(self.pred_test_actual, self.targ_test_actual)
        self.test_loglik = _test_loglik(self.pred_test, self.y, self.output_sigma)


        self.assign_optimizer(learning_rate)
        self.assign_session()
        self.make_metrics()

    def _prediction(self, inputs, nn, no_samples):
        return self._prediction_layer(inputs, nn, no_samples)

        # this samples a layer at a time
    def _prediction_layer(self, inputs, nn, no_samples):
        K = no_samples
        N = tf.shape(inputs)[0]

        with tf.name_scope('Expand_sample/'):
            act = tf.tile(tf.expand_dims(inputs, 0), [K, 1, 1])

        prev_acts = [act]

        for lidx in range(1, nn.num_internal_layers+1):

            plist = get_layer_parents(nn.conn_mat.keys(), lidx)
            # get parent layer output sizes and sum
            parent_acts = [prev_acts[i] for i in plist]

            dout = nn.num_units_in_each_layer[lidx]
            if dout == None: dout = 1
            layer_label = nn.layer_labels[lidx]
            activation = activation_dict[layer_label]

            with tf.name_scope(f'layer_{lidx}_{layer_label}_{dout}/'):

                act = tf.concat(parent_acts, -1)

                m_pre = tf.einsum('kni,io->kno', act, self.W_m[lidx-1])
                v_pre = tf.einsum('kni,io->kno', act ** 2.0, tf.exp(self.W_v[lidx-1]))
                eps_w = tf.random_normal([K, N, dout], 0.0, 1.0, dtype=tf.float32)
                pre_W = eps_w * tf.sqrt(1e-9 + v_pre) + m_pre
                eps_b = tf.random_normal([K, 1, dout], 0.0, 1.0, dtype=tf.float32)
                pre_b = eps_b * tf.exp(0.5 * self.b_v[lidx-1]) + self.b_m[lidx-1]
                pre = pre_W + pre_b
                if activation is not None:
                    act = activation(pre)
                else:
                    act = pre
                prev_acts.append(act)

        lidx += 1

        layer_label = nn.layer_labels[lidx]

        with tf.name_scope(f'layer_{lidx}_{layer_label}/'):

            plist = get_layer_parents(nn.conn_mat.keys(), lidx)
            # get parent layer output sizes and sum
            parent_acts = [prev_acts[i] for i in plist]

            scalar_mult = tf.Variable(1. / len(plist), dtype=tf.float32, trainable=False)  ### NEED TO VERIFY FLOAT 32
            act = tf.scalar_mul(scalar_mult, tf.add_n(parent_acts))

            # dout = 1
            # m_pre = tf.einsum('kni,io->kno', act, self.W_m[-1])
            # v_pre = tf.einsum('kni,io->kno', act ** 2.0, tf.exp(self.W_v[-1]))
            # eps_w = tf.random_normal([K, N, dout], 0.0, 1.0, dtype=tf.float32)
            # pre_W = eps_w * tf.sqrt(1e-9 + v_pre) + m_pre
            # eps_b = tf.random_normal([K, 1, dout], 0.0, 1.0, dtype=tf.float32)
            # pre_b = eps_b * tf.exp(0.5 * self.b_v[-1]) + self.b_m[-1]
            # pre = pre_W + pre_b

        return act

    def _KL_term(self):
        with tf.name_scope('kl'):
            kl = 0
            for i in range(len(self.W_m)):
                with tf.name_scope(f'layer_{i}'):

                    m, v = self.W_m[i], self.W_v[i]
                    m0, v0 = self.prior_W_m[i], self.prior_W_v[i]

                    prior = distributions.Normal(m0, tf.sqrt(v0))
                    weights = distributions.Normal(m, tf.exp(0.5 * v))
                    kl += tf.reduce_sum(distributions.kl_divergence(weights, prior))

                    m, v = self.b_m[i], self.b_v[i]
                    m0, v0 = self.prior_b_m[i], self.prior_b_v[i]

                    prior = distributions.Normal(m0, tf.sqrt(v0))
                    weights = distributions.Normal(m, tf.exp(0.5 * v))
                    kl += tf.reduce_sum(distributions.kl_divergence(weights, prior))

            return kl

    def create_weights_prior(self, input, nn, prior_mean, prior_var):

        W_m_w = []
        b_m_w = []
        W_v_w = []
        b_v_w = []

        W_m_p = []
        b_m_p = []
        W_v_p = []
        b_v_p = []

        douts = [input.get_shape().as_list()[-1]]

        for lidx in range(1, nn.num_internal_layers+1):
            plist = get_layer_parents(nn.conn_mat.keys(), lidx)
            # get parent layer output sizes and sum
            parent_outs = [douts[i] for i in plist]
            din = sum(parent_outs)
            # get number of units in layer

            dout = nn.num_units_in_each_layer[lidx]
            if dout == None: dout = 1
            douts.append(dout)

            layer_label = nn.layer_labels[lidx]
            activation = activation_dict[layer_label]

            with tf.name_scope(f'layer_{lidx}_{layer_label}_{dout}/'):
                # make weights
                Wi_m_val = tf.truncated_normal([din, dout], stddev=0.1)
                bi_m_val = tf.truncated_normal([dout], stddev=0.1)
                Wi_v_val = tf.constant(-11.0, shape=[din, dout])
                bi_v_val = tf.constant(-11.0, shape=[dout])

                Wi_m = tf.Variable(Wi_m_val, name='W_mean')
                bi_m = tf.Variable(bi_m_val, name='b_mean')
                Wi_v = tf.Variable(Wi_v_val, name='W_var')
                bi_v = tf.Variable(bi_v_val, name='b_var')

                W_m_w.append(Wi_m)
                b_m_w.append(bi_m)
                W_v_w.append(Wi_v)
                b_v_w.append(bi_v)

                tf.summary.histogram("weights_mean", Wi_m)
                tf.summary.histogram("bias_mean", bi_m)
                tf.summary.histogram("weights_variance", Wi_v)
                tf.summary.histogram("bais_variance", bi_v)

                # make prior

                Wi_m = prior_mean
                bi_m = prior_mean
                Wi_v = prior_var
                bi_v = prior_var

                W_m_p.append(Wi_m)
                b_m_p.append(bi_m)
                W_v_p.append(Wi_v)
                b_v_p.append(bi_v)

        return [W_m_w, b_m_w], [W_v_w, b_v_w], [W_m_p, b_m_p], [W_v_p, b_v_p]



class BayesMLPRegressionHyperprior(Reg_NN):
    def __init__(self, input_size, hidden_size, output_size, training_size, y_mu, y_sigma,
                 no_train_samples=10, no_pred_samples=100, prev_means=None, prev_log_variances=None,
                 learning_rate=0.001, prior_mean=0., prior_var=1., initial_output_noise=np.exp(-6), hyperprior=False):

        super(BayesMLPRegressionHyperprior, self).__init__(input_size, hidden_size, output_size, training_size, y_mu, y_sigma)

        self.hidden_size = deepcopy(hidden_size)
        self.learning_rate = learning_rate
        self.prior_var = prior_var
        self.sizes = [input_size] + deepcopy(hidden_size) + [output_size]
        self.hyperprior = hyperprior

        self.config = {
            'hidden_size': self.hidden_size,
            'learning_rate': self.learning_rate,
            'prior_var': self.prior_var,
            'hyper_prior': self.hyperprior
        }

        self.create_parameters()

        self.no_layers = len(self.sizes)-1
        self.no_train_samples = no_train_samples
        self.no_pred_samples = no_pred_samples
        self.training_size = training_size

        self.output_log_variance = tf.Variable(initial_value=np.log(initial_output_noise, dtype=np.float32), name='log_noise_variance')
        self.output_sigma = tf.exp(0.5 * self.output_log_variance)


        # def train predictions and training metric production
        self.pred_train = self._prediction(self.x, self.no_train_samples)

        self.pred_train_actual = self.pred_train * self.y_sigma + self.y_mu
        self.targ_train_actual = self.y * self.y_sigma + self.y_mu

        self.loglik = _loglik(self.pred_train_actual, self.targ_train_actual, self.output_sigma)
        self.KL = tf.div(self._KL_term(), self.training_size)

        self.cost = -self.loglik + self.KL


        # def test predictions and testing metrics
        self.pred_test  = self._prediction(self.x, self.no_pred_samples)

        self.pred_test_actual = self.pred_test * self.y_sigma + self.y_mu
        self.targ_test_actual = self.y * self.y_sigma + self.y_mu

        self.mse = _mse(self.pred_test_actual, self.targ_test_actual)
        self.test_loglik = _test_loglik(self.pred_test_actual, self.targ_test_actual, self.output_sigma)


        self.assign_optimizer(learning_rate)
        self.assign_session()
        self.make_metrics()

    def _prediction(self, inputs, no_samples):
        return self._prediction_layer(inputs, no_samples)

    def _prediction_layer(self, inputs, no_samples):
        with tf.name_scope('Expand_sample'):
            K = no_samples
            N = tf.shape(inputs)[0]
            act = tf.tile(tf.expand_dims(inputs, 0), [K, 1, 1])

        with tf.name_scope('model/'):
            for i in range(self.no_layers - 1):
                with tf.name_scope(f'layer_{i}/'):
                    din = self.sizes[i]
                    dout = self.sizes[i + 1]

                    W = self.W[i].value
                    b = self.b[i].value

                    m_pre = tf.einsum('kni,io->kno', act, W.loc)
                    v_pre = tf.einsum('kni,io->kno', act ** 2.0, W.scale ** 2)
                    eps_w = tf.random_normal([K, N, dout], 0.0, 1.0, dtype=tf.float32)
                    pre_W = eps_w * tf.sqrt(1e-9 + v_pre) + m_pre
                    eps_b = tf.random_normal([K, 1, dout], 0.0, 1.0, dtype=tf.float32)
                    pre_b = eps_b * b.scale + b.loc
                    pre = pre_W + pre_b
                    act = tf.nn.relu(pre)

            with tf.name_scope(f'layer_{self.no_layers-1}/'):
                din = self.sizes[-2]
                dout = self.sizes[-1]

                W = self.W[-1].value
                b = self.b[-1].value

                m_pre = tf.einsum('kni,io->kno', act, W.loc)
                v_pre = tf.einsum('kni,io->kno', act ** 2.0, W.scale ** 2)
                eps_w = tf.random_normal([K, N, dout], 0.0, 1.0, dtype=tf.float32)
                pre_W = eps_w * tf.sqrt(1e-9 + v_pre) + m_pre
                eps_b = tf.random_normal([K, 1, dout], 0.0, 1.0, dtype=tf.float32)
                pre_b = eps_b * b.scale + b.loc
                pre = pre_W + pre_b

                return pre

    def _KL_term(self):
        with tf.name_scope('kl'):
            kl = 0
            for i, (weight, bias) in enumerate(zip(self.W, self.b)):
                with tf.name_scope(f'layer_{i}'):
                    kl += weight.KL()
                    kl += bias.KL()
            return kl

    def create_parameters(self):
        self.W = []; self.b = []
        with tf.name_scope('model/'):
            for i, (din, dout) in enumerate(zip(self.sizes[:-1], self.sizes[1:])):
                with tf.name_scope(f'layer_{i}/'):
                    self.W.append(make_weight_parameter([din, dout], self.prior_var, self.hyperprior))
                    self.b.append(make_weight_parameter([dout], self.prior_var, self.hyperprior))