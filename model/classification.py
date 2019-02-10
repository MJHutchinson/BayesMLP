import tensorflow as tf
import numpy as np
from copy import deepcopy

from utils.nn_utils import activation_dict

np.random.seed(0)
tf.set_random_seed(0)

def get_layer_parents(adjList, lidx):
    """ Returns parent layer indices for a given layer index. """
    # Return all parents (layer indices in list) for layer lidx
    return [e[0] for e in adjList if e[1] == lidx]


def _loglik(predictions, targets):
    with tf.name_scope('loglik'):
        pred = predictions
        targets = tf.tile(tf.expand_dims(targets, 0), [predictions.get_shape().as_list()[0], 1, 1])
        log_lik = - tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred, labels=targets))
        return log_lik


def _test_loglik(predictions, targets):
    with tf.name_scope('test_loglik'):
        pred = tf.reduce_mean(predictions, axis=0)
        log_lik = - tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred, labels=targets))
        return log_lik


def _predict_correct(predictions, y_test):
    pred_classes = tf.argmax(tf.reduce_mean(predictions, axis=0), axis=1)
    y_classes = tf.argmax(y_test, axis=1)
    correct = tf.equal(pred_classes, y_classes)
    correct = tf.reduce_sum(tf.cast(correct, tf.float32))
    return correct


class Cla_NN(object):
    def __init__(self, input_size, output_size):
        # input and output placeholders
        self.x = tf.placeholder(tf.float32, [None, input_size])
        self.y = tf.placeholder(tf.float32, [None, output_size])

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
            config = tf.ConfigProto()  # log_device_placement=True
            config.gpu_options.allow_growth = True

            # launch a session
            self.sess = tf.Session(config=config)
            self.sess.run(init)

    def train(self, x_train, y_train, no_epochs=100, batch_size=100, display_epoch=5):

        costs = []
        # Training cycle
        for epoch in range(no_epochs):
            avg_cost, avg_kl, avg_ll = self.train_one(x_train, y_train)
            # Display logs per epoch step
            if epoch % display_epoch == 0:
                print("Epoch:", '%04d' % (epoch + 1), "cost=",
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
                [self.train_step, self.cost, self.KL, self.train_loglik],
                feed_dict={self.x: batch_x, self.y: batch_y})
            # Compute average loss
            avg_cost += c / total_batch
            avg_kl += kl / total_batch
            avg_ll += ll / total_batch

        return avg_cost, avg_kl, avg_ll

    def train_one_optim_step(self, dataloader):
        batch_x, batch_y = dataloader.next_train_batch()

        _, c, kl, ll = self.sess.run(
            [self.train_step, self.cost, self.KL, self.train_loglik],
            feed_dict={self.x: batch_x, self.y: batch_y})

        return c, kl, ll

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

    def prediction_prob(self, x_test):
        prob = self.sess.run([tf.nn.softmax(self.pred_test)], feed_dict={self.x: x_test})[0]
        return prob

    def prediction_class(self, x_test):
        classes = self.sess.run([tf.argmax(tf.reduce_mean(tf.nn.softmax(self.pred_test), axis=0), axis=1)], feed_dict={self.x: x_test})[0]
        return classes

    def accuracy(self, x_test, y_test, batch_size=100):
        sess = self.sess

        N = x_test.shape[0]
        if batch_size > N:
            batch_size = N

        perm_inds = list(range(x_test.shape[0]))
        np.random.shuffle(perm_inds)
        cur_x_train = x_test[perm_inds]
        cur_y_train = y_test[perm_inds]

        correct = 0.
        total = 0.
        loglik = 0.
        total_batch = int(np.ceil(N * 1.0 / batch_size))
        # Loop over all batches
        for i in range(total_batch):
            start_ind = i * batch_size
            end_ind = np.min([(i + 1) * batch_size, N])
            batch_x = cur_x_train[start_ind:end_ind, :]
            batch_y = cur_y_train[start_ind:end_ind, :]
            # Run optimization op (backprop) and cost op (to get loss value)
            corr, ll  = sess.run(
                [self.corr_test, self.test_loglik],
                feed_dict={self.x: batch_x, self.y: batch_y})
            # Compute average loss

            fraction = (end_ind - start_ind) / N

            correct += corr / N
            loglik += ll * fraction

        return correct, ll


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
            self.test_acc      = tf.placeholder(tf.float32, shape=None, name='test_rmse_summary')

            self.train_logloss_true = tf.placeholder(tf.float32, shape=None, name='train_logloss_summary_true')
            self.test_logloss_true  = tf.placeholder(tf.float32, shape=None, name='test_logloss_summary_true')
            self.output_sigma_true  = tf.placeholder(tf.float32, shape=None, name='homoskedastic_noise_true')

            train_cost_summary    = tf.summary.scalar('train_cost',     self.train_cost, family='transforms_space')
            train_logloss_summary = tf.summary.scalar('train_logloss',  self.train_logloss, family='transforms_space')
            train_kl_summary      = tf.summary.scalar('train_kl',       self.train_kl, family='real_space')
            test_logloss_summary  = tf.summary.scalar('test_logloss',   self.test_logloss, family='transforms_space')

            train_logloss_summary_true = tf.summary.scalar('train_logloss_true',  self.train_logloss_true, family='real_space')
            test_logloss_summary_true  = tf.summary.scalar('test_logloss_true',   self.test_logloss_true, family='real_space')
            test_acc_summary           = tf.summary.scalar('test_acc',      self.test_acc, family='real_space')

            self.performance_metrics = tf.summary.merge_all()

    def log_metrics(self, train_cost, train_logloss, train_kl, test_logloss, test_acc, train_logloss_true, test_logloss_true):
        return self.sess.run(self.performance_metrics, feed_dict={
            self.train_cost:train_cost,
            self.train_logloss: train_logloss,
            self.train_kl: train_kl,
            self.test_logloss: test_logloss,

            self.train_logloss_true: train_logloss_true,
            self.test_logloss_true: test_logloss_true,
            self.test_acc: test_acc
        })


""" Neural Network MLP Model """


""" Bayesian MLP Model """

class BayesMLPClassification(Cla_NN):
    def __init__(self, input_size, hidden_size, output_size, training_size,
                 no_train_samples=10, no_pred_samples=100, prev_means=None, prev_log_variances=None,
                 learning_rate=0.001,
                 prior_mean=0, prior_var=1, activation=tf.nn.relu):
        super(BayesMLPClassification, self).__init__(input_size, output_size)

        self.hidden_sizes = hidden_size
        self.prior_var = prior_var
        self.learning_rate = learning_rate
        self.input_size = input_size
        self.output_size = output_size

        self.config = {
            'hidden_sizes': self.hidden_sizes,
            'learning_rate': self.learning_rate,
            'prior_var': self.prior_var,
        }

        m, v, self.size = self.create_weights(input_size, hidden_size, output_size, prev_means, prev_log_variances)
        self.W_m, self.b_m = m[0], m[1]
        self.W_v, self.b_v = v[0], v[1]
        self.weights = [m, v]

        self.W = self.W_m + self.W_v
        self.b = self.b_m + self.b_v

        m, v = self.create_prior(input_size, hidden_size, output_size, prev_means, prev_log_variances, prior_mean, prior_var)
        self.prior_W_m, self.prior_b_m = m[0], m[1]
        self.prior_W_v, self.prior_b_v = v[0], v[1]

        self.no_layers = len(self.size)-1

        self.no_layers = len(self.size) - 1
        self.no_train_samples = no_train_samples
        self.no_pred_samples = no_pred_samples

        self.activation = activation

        self.pred_train = self._prediction(self.x, self.no_train_samples)
        self.pred_test = self._prediction(self.x, self.no_pred_samples)

        self.corr_train = self._predict_correct(self.pred_train, self.y)
        self.corr_test = self._predict_correct(self.pred_test, self.y)

        self.KL = tf.div(self._KL_term(), training_size)
        self.train_loglik = - self._loglik(self.pred_train, self.y)
        self.test_loglik = - self._test_loglik(self.pred_test, self.y)
        self.cost = self.KL + self.train_loglik

        self.assign_optimizer(learning_rate)
        self.assign_session()
        self.make_metrics()

    def _predict_correct(self, y_pred, y_test):
        pred_classes = tf.argmax(tf.reduce_mean(tf.nn.softmax(y_pred), axis=0), axis=1)
        y_classes = tf.argmax(y_test, axis=1)
        correct = tf.equal(pred_classes, y_classes)
        correct = tf.reduce_sum(tf.cast(correct, tf.float32))
        return correct

    def _prediction(self, inputs, no_samples):
        return self._prediction_layer(inputs, no_samples)

        # this samples a layer at a time
    def _prediction_layer(self, inputs, no_samples):
        K = no_samples
        N = tf.shape(inputs)[0]
        act = tf.tile(tf.expand_dims(inputs, 0), [K, 1, 1])

        # for i in range(self.no_layers - 1):
        #     din = self.size[i]
        #     dout = self.size[i + 1]
        #     eps_w = tf.random_normal((K, din, dout), 0, 1, dtype=tf.float32)
        #     eps_b = tf.random_normal((K, 1, dout), 0, 1, dtype=tf.float32)
        #
        #     weights = tf.add(tf.multiply(eps_w, tf.exp(0.5 * self.W_v[i])), self.W_m[i])
        #     biases = tf.add(tf.multiply(eps_b, tf.exp(0.5 * self.b_v[i])), self.b_m[i])
        #     pre = tf.add(tf.einsum('kni,kio->kno', act, weights), biases)
        #     act = tf.nn.relu(pre)
        #
        # din = self.size[-2]
        # dout = self.size[-1]
        # eps_w = tf.random_normal((K, din, dout), 0, 1, dtype=tf.float32)
        # eps_b = tf.random_normal((K, 1, dout), 0, 1, dtype=tf.float32)
        #
        # weights = tf.add(tf.multiply(eps_w, tf.exp(0.5 * self.W_v[-1])), self.W_m[-1])
        # biases = tf.add(tf.multiply(eps_b, tf.exp(0.5 * self.b_v[-1])), self.b_m[-1])
        #
        # act = tf.expand_dims(act, 3)
        # weights = tf.expand_dims(weights, 1)
        #
        # pre = tf.add(tf.reduce_sum(act * weights, 2), biases)
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
                    # act = tf.nn.relu(pre)
                    act = self.activation(pre)

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

    # computes log likelihood of input for against target
    def _loglik(self, preds, targets):
        with tf.name_scope('loglik'):
            pred = preds
            targets = tf.tile(tf.expand_dims(targets, 0), [self.no_train_samples, 1, 1])
            log_lik = - tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred, labels=targets))
            return log_lik

    def _test_loglik(self, preds, targets):
        with tf.name_scope('test_loglik'):
            preds = tf.nn.softmax(preds)
            pred = tf.reduce_mean(preds, axis=0)
            log_lik = - tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred, labels=targets))
            return log_lik

    def _KL_term(self):
        with tf.name_scope('kl'):
            kl = 0
            for i in range(self.no_layers):
                with tf.name_scope(f'layer_{i}'):
                    din = self.size[i]
                    dout = self.size[i + 1]
                    m, v = self.W_m[i], self.W_v[i]
                    m0, v0 = self.prior_W_m[i], self.prior_W_v[i]
                    const_term = -0.5 * dout * din
                    log_std_diff = 0.5 * tf.reduce_sum(np.log(v0) - v)
                    mu_diff_term = 0.5 * tf.reduce_sum((tf.exp(v) + (m0 - m) ** 2) / v0)
                    kl += const_term + log_std_diff + mu_diff_term

                    m, v = self.b_m[i], self.b_v[i]
                    m0, v0 = self.prior_b_m[i], self.prior_b_v[i]
                    const_term = -0.5 * dout
                    log_std_diff = 0.5 * tf.reduce_sum(np.log(v0) - v)
                    mu_diff_term = 0.5 * tf.reduce_sum((tf.exp(v) + (m0 - m) ** 2) / v0)
                    kl += const_term + log_std_diff + mu_diff_term

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
                        Wi_v_val = tf.constant(-6.0, shape=[din, dout])
                        bi_v_val = tf.constant(-6.0, shape=[dout])
                    else:
                        Wi_m_val = prev_weights[0][i]
                        bi_m_val = prev_weights[1][i]
                        if prev_variances is None:
                            Wi_v_val = tf.constant(-6.0, shape=[din, dout])
                            bi_v_val = tf.constant(-6.0, shape=[dout])
                        else:
                            Wi_v_val = prev_variances[0][i]
                            bi_v_val = prev_variances[1][i]

                    Wi_m = tf.Variable(Wi_m_val)
                    bi_m = tf.Variable(bi_m_val)
                    Wi_v = tf.Variable(Wi_v_val)
                    bi_v = tf.Variable(bi_v_val)
                    W_m.append(Wi_m)
                    b_m.append(bi_m)
                    W_v.append(Wi_v)
                    b_v.append(bi_v)

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


class BayesMLPNNClassification(Cla_NN):
    def __init__(self, input_size, nn, output_size, training_size,
                 no_train_samples=10, no_pred_samples=100, prev_means=None, prev_log_variances=None,
                 learning_rate=0.001, prior_mean=0., prior_var=1., hyperprior=False, **kwargs):

        super(BayesMLPNNClassification, self).__init__(input_size, output_size)

        self.conn_mat = nn.conn_mat
        self.hidden_sizes = nn.num_units_in_each_layer
        self.layer_labels = nn.layer_labels
        self.prior_var = prior_var
        self.learning_rate = learning_rate
        self.hyperprior = hyperprior
        self.input_size = input_size
        self.output_size = output_size

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

        self.pred_train = self._prediction(self.x, self.no_train_samples)
        self.pred_test  = self._prediction(self.x, self.no_pred_samples)
        self.corr_train = _predict_correct(self.pred_train, self.y)
        self.corr_test  = _predict_correct(self.pred_test,  self.y)

        self.KL = tf.div(self._KL_term(), training_size)
        self.train_loglik = - _loglik(self.pred_train, self.y)
        self.test_loglik  = - _test_loglik(self.pred_test, self.y)
        self.cost = self.KL + self.train_loglik

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
            if dout == None: dout = self.output_size
            douts.append(dout)

            layer_label = self.nn.layer_labels[lidx]

            with tf.name_scope(f'layer_{lidx}_{layer_label}_{dout}/'):
                self.W.append(make_weight_parameter([din, dout], self.prior_var, self.hyperprior))
                self.b.append(make_weight_parameter([dout], self.prior_var, self.hyperprior))
