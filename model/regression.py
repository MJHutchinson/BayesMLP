import os
import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow_probability.python.distributions as distributions
import numpy as np
from copy import deepcopy

np.random.seed(0)
# tf.set_random_seed(0)

os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
tf.logging.set_verbosity(tf.logging.WARN)

INITIAL_LOG_NOISE = -6.

activation_dict = {'relu':tf.nn.relu,
                'elu':tf.nn.elu,
                'crelu':tf.nn.crelu,
                'relu6':tf.nn.relu6,
                'softplus':tf.nn.softplus,
                'softmax':tf.nn.softmax,
                'linear':None,
                'logistic':tf.nn.sigmoid,
                'tanh':tf.nn.tanh,
                'leaky-relu':tf.nn.relu, # Need to update tf for leaky_relu; leaky_relu --> relu.
                'relu-x':tf.nn.relu, # Not sure what relu-x is in tf; relu-x --> relu.
                'step':tf.nn.tanh, # Not sure how to do step in tf; step --> tanh
                }

def get_layer_parents(adjList, lidx):
    """ Returns parent layer indices for a given layer index. """
    # Return all parents (layer indices in list) for layer lidx
    return [e[0] for e in adjList if e[1] == lidx]


def _mse(preds, targets):
    with tf.name_scope('mse'):
        mse = tf.reduce_mean(tf.squared_difference(preds, targets))
        return mse


def _loglik(pred, targets, noise_std):
    with tf.name_scope('loglik'):
        # se = tf.squared_difference(pred, targets)
        # const_term = - 0.5 * tf.log(tf.constant(2 * np.pi))
        # noise_term = - 0.5 * self.noise_var
        # se_norm_term = - tf.reduce_mean(se/(2* tf.exp(self.noise_var)))
        # return noise_term + se_norm_term + const_term
        log_probs = distributions.Normal(loc=targets, scale=noise_std).log_prob(pred)
        return tf.reduce_mean(log_probs)


def _test_loglik(pred, targets, noise_std):
    with tf.name_scope('test_loglik'):
        # se = tf.squared_difference(pred, targets)
        # probs = tf.div(1., tf.sqrt(tf.constant(2 * np.pi) * tf.exp(log_variance))) * tf.exp(-tf.div(se, 2 * tf.exp(log_variance)))
        # probs = tf.reduce_mean(probs, axis=0)
        # log_probs = tf.log(probs)
        # return tf.reduce_mean(log_probs)
        log_probs = distributions.Normal(loc=targets, scale=noise_std).log_prob(pred)
        log_probs = tf.reduce_logsumexp(log_probs, axis=0) - tf.log(tf.to_float(tf.shape(log_probs)[0]))
        return tf.reduce_mean(log_probs)


def _KL_term(weights, prior):
        W_m, W_v = weights[0][0], weights[1][0]
        b_m, b_v = weights[0][1], weights[1][1]

        prior_W_m, prior_W_v = prior[0][0], prior[1][0]
        prior_b_m, prior_b_v = prior[0][1], prior[1][1]

        with tf.name_scope('kl'):
            kl = 0
            for i in range(len(W_m)):
                with tf.name_scope(f'layer_{i}'):

                    m, v = W_m[i], W_v[i]
                    m0, v0 = prior_W_m[i], prior_W_v[i]

                    prior_dist = distributions.Normal(m0, tf.sqrt(v0))
                    weights_dist = distributions.Normal(m, tf.exp(0.5 * v))
                    kl += tf.reduce_sum(distributions.kl_divergence(weights_dist, prior_dist))

                    m, v = b_m[i], b_v[i]
                    m0, v0 = prior_b_m[i], prior_b_v[i]

                    prior_dist = distributions.Normal(m0, tf.sqrt(v0))
                    weights_dist = distributions.Normal(m, tf.exp(0.5 * v))
                    kl += tf.reduce_sum(distributions.kl_divergence(weights_dist, prior_dist))

            return kl


class Reg_NN(object):
    def __init__(self, input_size, hidden_size, output_size, training_size):
        # input and output placeholders
        self.x = tf.placeholder(tf.float32, [None, input_size], name='x')
        self.y = tf.placeholder(tf.float32, [None, output_size], name='y')

        np.random.seed(1)
        tf.random.set_random_seed(1)


    def assign_optimizer(self, learning_rate=0.001):
        with tf.name_scope('optimiser'):
            self.train_step = tf.train.AdamOptimizer(learning_rate).minimize(self.cost)

    def assign_session(self):
        # Initializing the variables
        with tf.name_scope('initialisation'):
            init = tf.global_variables_initializer()
            init2 = tf.local_variables_initializer()
            config = tf.ConfigProto() # log_device_placement=True
            config.gpu_options.allow_growth=True


            # launch a session
            self.sess = tf.Session(config=config)
            self.sess.run([init, init2])

    def train(self, x_train, y_train, no_epochs=100, batch_size=100, display_epoch=5):

        costs = []
        # Training cycle
        for epoch in range(no_epochs):
            avg_cost, avg_kl, avg_ll = self.train_one(x_train, y_train)
            # Display logs per epoch step
            if epoch % display_epoch == 0:
                print("Epoch:", '%04d' % (epoch + 1), "cost=", \
                      "{:.9f} ".format(avg_cost))
            costs.append(avg_cost)
        print("Optimization Finished!")
        return costs

    def train_one(self, x_train, y_train, batch_size=100):
        sess = self.sess

        N = x_train.shape[0]
        if batch_size > N:
            batch_size = N

        perm_inds = list(range(x_train.shape[0]))
        np.random.shuffle(perm_inds)
        cur_x_train = x_train[perm_inds]
        cur_y_train = y_train[perm_inds]

        avg_cost = 0.
        avg_kl = 0.
        avg_ll = 0.
        total_batch = int(np.ceil(N * 1.0 / batch_size))
        # Loop over all batches
        for i in range(total_batch):
            start_ind = i * batch_size
            end_ind = np.min([(i + 1) * batch_size, N])
            batch_x = cur_x_train[start_ind:end_ind, :]
            batch_y = cur_y_train[start_ind:end_ind, :]
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c, kl, ll = sess.run(
                [self.train_step, self.cost, self.KL, self.loglik],
                feed_dict={self.x: batch_x, self.y: batch_y})
            # Compute average loss
            fraction = (end_ind - start_ind) / N
            avg_cost += c * fraction
            avg_kl += kl * fraction
            avg_ll += ll * fraction

        return avg_cost, avg_kl, avg_ll

    def prediction(self, x_test, batch_size=100):
        sess = self.sess

        N = x_test.shape[0]
        if batch_size > N:
            batch_size = N

        preds = None
        total_batch = int(np.ceil(N * 1.0 / batch_size))
        # Loop over all batches
        for i in range(total_batch):
            start_ind = i * batch_size
            end_ind = np.min([(i + 1) * batch_size, N])
            batch_x = x_test[start_ind:end_ind, :]
            # Run optimization op (backprop) and cost op (to get loss value)
            pred = sess.run(
                [self.pred_test],
                feed_dict={self.x: batch_x})[0]
            # Compute average loss
            if preds is None:
                preds = pred
            else:
                preds = np.append(preds, pred, axis=1)

        return preds

    def accuracy(self, x_test, y_test, batch_size=100):
        sess = self.sess

        N = x_test.shape[0]
        if batch_size > N:
            batch_size = N

        perm_inds = list(range(x_test.shape[0]))
        np.random.shuffle(perm_inds)
        cur_x_test = x_test[perm_inds]
        cur_y_test = y_test[perm_inds]

        ll = 0.
        mse = 0.
        total_batch = int(np.ceil(N * 1.0 / batch_size))
        # Loop over all batches
        for i in range(total_batch):
            start_ind = i * batch_size
            end_ind = np.min([(i + 1) * batch_size, N])
            batch_x = cur_x_test[start_ind:end_ind, :]
            batch_y = cur_y_test[start_ind:end_ind, :]
            # Run optimization op (backprop) and cost op (to get loss value)
            _ll, _mse = sess.run(
                [self.test_loglik, self.mse],
                feed_dict={self.x: batch_x, self.y: batch_y})

            # Compute average loss
            fraction = (end_ind - start_ind) / N
            ll += _ll * fraction
            mse += _mse * fraction

        return ll, np.sqrt(mse)

    def get_weights(self):
        weights = self.sess.run([self.weights])[0]
        return weights

    def close_session(self):
        self.sess.close()

    def get_config(self):
        return self.config

    def __str__(self):
        config = self.get_config()
        s = ''
        for key in config:
            s += f'{key}_{config[key]}_'

        s = s[:-1]
        s = s.replace('\n', ' ').replace('\t1.0', ' ')#.replace(' ', '')
        return s

    def make_metrics(self):
        with tf.name_scope('performance'):
            self.train_cost    = tf.placeholder(tf.float32, shape=None, name='train_cost_summary')
            self.train_logloss = tf.placeholder(tf.float32, shape=None, name='train_logloss_summary')
            self.train_kl      = tf.placeholder(tf.float32, shape=None, name='train_kl_summary')
            self.test_logloss  = tf.placeholder(tf.float32, shape=None, name='test_logloss_summary')
            self.test_rmse     = tf.placeholder(tf.float32, shape=None, name='test_rmse_summary')

            self.train_logloss_true = tf.placeholder(tf.float32, shape=None, name='train_logloss_summary_true')
            self.test_logloss_true  = tf.placeholder(tf.float32, shape=None, name='test_logloss_summary_true')
            self.test_rmse_true     = tf.placeholder(tf.float32, shape=None, name='test_rmse_summary_true')
            self.output_sigma_true  = tf.placeholder(tf.float32, shape=None, name='homoskedastic_noise_true')

            train_cost_summary    = tf.summary.scalar('train_cost',     self.train_cost, family='transforms_space')
            train_logloss_summary = tf.summary.scalar('train_logloss',  self.train_logloss, family='transforms_space')
            train_kl_summary      = tf.summary.scalar('train_kl',       self.train_kl, family='real_space')
            test_logloss_summary  = tf.summary.scalar('test_logloss',   self.test_logloss, family='transforms_space')
            test_rmse_summary     = tf.summary.scalar('test_rmse',      self.test_rmse, family='transforms_space')
            noise_output_noise_summary = tf.summary.scalar('homoskedastic_noise', self.output_sigma, family='transforms_space')

            train_logloss_summary_true = tf.summary.scalar('train_logloss_true',  self.train_logloss_true, family='real_space')
            test_logloss_summary_true  = tf.summary.scalar('test_logloss_true',   self.test_logloss_true, family='real_space')
            test_rmse_summary_true     = tf.summary.scalar('test_rmse_true',      self.test_rmse_true, family='real_space')
            noise_output_noise_summary_true = tf.summary.scalar('homoskedastic_noise_true', self.output_sigma_true, family='real_space')

            self.performance_metrics = tf.summary.merge_all()

    def log_metrics(self, train_cost, train_logloss, train_kl, test_logloss, test_rmse, train_logloss_true, test_logloss_true, test_rmse_true, output_sigma_true):
        return self.sess.run(self.performance_metrics, feed_dict={
            self.train_cost:train_cost,
            self.train_logloss: train_logloss,
            self.train_kl: train_kl,
            self.test_logloss: test_logloss,
            self.test_rmse: test_rmse,

            self.train_logloss_true: train_logloss_true,
            self.test_logloss_true: test_logloss_true,
            self.test_rmse_true: test_rmse_true,
            self.output_sigma_true: output_sigma_true
        })

""" Neural Network MLP Model """


class BayesMLPRegression(Reg_NN):
    def __init__(self, input_size, hidden_size, output_size, training_size,
                 no_train_samples=10, no_pred_samples=100, prev_means=None, prev_log_variances=None,
                 learning_rate=0.001, prior_mean=0., prior_var=1., initial_output_noise=np.exp(-6)):

        super(BayesMLPRegression, self).__init__(input_size, hidden_size, output_size, training_size)

        self.config = {
            'hidden_size': hidden_size,
            'learning_rate': learning_rate,
            'prior_var': prior_var
        }

        m, v, self.size = self.create_weights(input_size, hidden_size, output_size, prev_means, prev_log_variances)
        self.W_m, self.b_m = m[0], m[1]
        self.W_v, self.b_v = v[0], v[1]
        self.weights = [m, v]

        m, v = self.create_prior(input_size, hidden_size, output_size, prev_means, prev_log_variances, prior_mean, prior_var)
        self.prior_W_m, self.prior_b_m = m[0], m[1]
        self.prior_W_v, self.prior_b_v = v[0], v[1]
        self.priors = [m, v]

        self.prior_mean = prior_mean
        self.prior_var = prior_var

        self.no_layers = len(self.size)-1
        self.no_train_samples = no_train_samples
        self.no_pred_samples = no_pred_samples
        self.training_size = training_size

        self.output_log_variance = tf.Variable(initial_value=np.log(initial_output_noise, dtype=np.float32), name='log_noise_variance')
        self.output_sigma = tf.exp(0.5 * self.output_log_variance)


        # def train predictions and training metric production
        self.pred_train = self._prediction(self.x, self.no_train_samples)

        self.loglik = _loglik(self.pred_train, self.y, self.output_sigma)
        self.KL = tf.div(_KL_term(self.weights, self.priors), self.training_size)

        self.cost = -self.loglik + self.KL


        # def test predictions and testing metrics
        self.pred_test  = self._prediction(self.x, self.no_pred_samples)

        self.mse = _mse(self.pred_test, self.y)
        self.test_loglik = _test_loglik(self.pred_test, self.y, self.output_sigma)


        self.assign_optimizer(learning_rate)
        self.assign_session()
        self.make_metrics()

    def _prediction(self, inputs, no_samples):
        return self._prediction_layer(inputs, no_samples)

        # this samples a layer at a time
    def _prediction_layer(self, inputs, no_samples):
        with tf.name_scope('Expand_sample'):
            K = no_samples
            N = tf.shape(inputs)[0]
            act = tf.tile(tf.expand_dims(inputs, 0), [K, 1, 1])


        with tf.device('/gpu:0'):
            with tf.name_scope('model/'):
                for i in range(self.no_layers - 1):
                    with tf.name_scope(f'layer_{i}/'):
                        din = self.size[i]
                        dout = self.size[i + 1]

                        m_pre = tf.einsum('kni,io->kno', act, self.W_m[i])
                        v_pre = tf.einsum('kni,io->kno', act ** 2.0, tf.exp(self.W_v[i]))
                        eps_w = tf.random_normal([K, N, dout], 0.0, 1.0, dtype=tf.float32)
                        pre_W = eps_w * tf.sqrt(1e-9 + v_pre) + m_pre
                        eps_b = tf.random_normal([K, 1, dout], 0.0, 1.0, dtype=tf.float32)
                        pre_b = eps_b * tf.exp(0.5 * self.b_v[i]) + self.b_m[i]
                        pre = pre_W + pre_b
                        act = tf.nn.relu(pre)

                with tf.name_scope(f'layer_{self.no_layers-1}/'):
                    din = self.size[-2]
                    dout = self.size[-1]

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
                    din = self.size[i]
                    dout = self.size[i + 1]

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

    def create_weights(self, in_dim, hidden_size, out_dim, prev_weights, prev_variances):
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

                    # tf.summary.histogram("weights_mean", Wi_m)
                    # tf.summary.histogram("bias_mean", bi_m)
                    # tf.summary.histogram("weights_variance", Wi_v)
                    # tf.summary.histogram("bais_variance", bi_v)

            return [W_m, b_m], [W_v, b_v], hidden_size


    def create_prior(self, in_dim, hidden_size, out_dim, prev_weights, prev_variances, prior_mean, prior_var):
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


""" Abstracted parameters network"""

from model.variational_parameter import make_weight_parameter

class BayesMLPNNRegression(Reg_NN):
    def __init__(self, input_size, nn, training_size,
                 no_train_samples=10, no_pred_samples=100, prev_means=None, prev_log_variances=None,
                 learning_rate=0.001, prior_mean=0., prior_var=1., hyperprior=False, **kwargs):

        super(BayesMLPNNRegression, self).__init__(input_size, 0, 1, training_size)

        self.conn_mat = nn.conn_mat
        self.hidden_sizes = nn.num_units_in_each_layer
        self.layer_labels = nn.layer_labels
        self.prior_var = prior_var
        self.learning_rate = learning_rate
        self.hyperprior = hyperprior

        self.config = {
            'hidden_sizes': self.hidden_sizes,
            'con_mat': self.conn_mat,
            'learning_rate': self.learning_rate,
            'prior_var': self.prior_var,
            'hyperprior': self.hyperprior
        }

        self.nn = nn

        self.create_parameters()

        self.no_train_samples = no_train_samples
        self.no_pred_samples = no_pred_samples
        self.training_size = training_size

        self.output_log_variance = tf.Variable(initial_value=INITIAL_LOG_NOISE, name='log_noise_variance')
        self.output_sigma = tf.exp(0.5 * self.output_log_variance)

        # def train predictions and training metric production
        self.pred_train = self._prediction(self.x, self.no_train_samples)

        self.loglik = _loglik(self.pred_train, self.y, self.output_sigma)
        self.KL = tf.div(self._KL_term(), self.training_size)

        self.cost = -self.loglik + self.KL

        # def test predictions and testing metrics
        self.pred_test = self._prediction(self.x, self.no_pred_samples)

        self.mse = _mse(self.pred_test, self.y)
        self.test_loglik = _test_loglik(self.pred_test, self.y, self.output_sigma)

        self.assign_optimizer(learning_rate)
        self.assign_session()
        self.make_metrics()

    def _prediction(self, inputs, no_samples):
        return self._prediction_layer(inputs, no_samples)

        # this samples a layer at a time
    def _prediction_layer(self, inputs, no_samples):
        K = no_samples
        N = tf.shape(inputs)[0]

        with tf.name_scope('Expand_sample/'):
            act = tf.tile(tf.expand_dims(inputs, 0), [K, 1, 1])

        prev_acts = [act]

        for lidx in range(1, self.nn.num_internal_layers+1):

            plist = get_layer_parents(self.nn.conn_mat.keys(), lidx)
            # get parent layer output sizes and sum
            parent_acts = [prev_acts[i] for i in plist]

            dout = self.nn.num_units_in_each_layer[lidx]
            if dout == None: dout = 1
            layer_label = self.nn.layer_labels[lidx]
            activation = activation_dict[layer_label]

            with tf.name_scope(f'layer_{lidx}_{layer_label}_{dout}/'):

                act = tf.concat(parent_acts, -1)

                W = self.W[lidx-1].value
                b = self.b[lidx-1].value

                m_pre = tf.einsum('kni,io->kno', act, W.loc)
                v_pre = tf.einsum('kni,io->kno', act ** 2.0, W.scale ** 2)
                eps_w = tf.random_normal([K, N, dout], 0.0, 1.0, dtype=tf.float32)
                pre_W = eps_w * tf.sqrt(1e-9 + v_pre) + m_pre
                eps_b = tf.random_normal([K, 1, dout], 0.0, 1.0, dtype=tf.float32)
                pre_b = eps_b * b.scale + b.loc
                pre = pre_W + pre_b

                if activation is not None:
                    act = activation(pre)
                else:
                    act = pre
                prev_acts.append(act)

        lidx += 1

        layer_label = self.nn.layer_labels[lidx]

        with tf.name_scope(f'layer_{lidx}_{layer_label}/'):

            plist = get_layer_parents(self.nn.conn_mat.keys(), lidx)
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
            for i, (weight, bias) in enumerate(zip(self.W, self.b)):
                with tf.name_scope(f'layer_{i}'):
                    kl += weight.KL()
                    kl += bias.KL()
            return kl

    def create_parameters(self):

        self.W = []; self.b = []

        douts = [self.x.get_shape().as_list()[-1]]

        for lidx in range(1, self.nn.num_internal_layers+1):
            plist = get_layer_parents(self.nn.conn_mat.keys(), lidx)
            # get parent layer output sizes and sum
            parent_outs = [douts[i] for i in plist]
            din = sum(parent_outs)
            # get number of units in layer

            dout = self.nn.num_units_in_each_layer[lidx]
            if dout == None: dout = 1
            douts.append(dout)

            layer_label = self.nn.layer_labels[lidx]

            with tf.name_scope(f'layer_{lidx}_{layer_label}_{dout}/'):
                self.W.append(make_weight_parameter([din, dout], self.prior_var, self.hyperprior))
                self.b.append(make_weight_parameter([dout], self.prior_var, self.hyperprior))
