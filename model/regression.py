import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow_probability.python.distributions as distributions
import numpy as np
from copy import deepcopy

np.random.seed(0)
tf.set_random_seed(0)


# variable initialization functions
def weight_variable(shape, init_weights=None):
    if init_weights is not None:
        initial = tf.constant(init_weights)
    else:
        initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def small_variable(shape):
    initial = tf.constant(-6.0, shape=shape)
    return tf.Variable(initial)


def zero_variable(shape):
    initial = tf.zeros(shape=shape)
    return tf.Variable(initial)


def _create_weights_mf(in_dim, hidden_size, out_dim, init_weights=None, init_variances=None):
    size = deepcopy(hidden_size)
    size.append(out_dim)
    size.insert(0, in_dim)
    no_params = 0
    for i in range(len(size) - 1):
        no_weights = size[i] * size[i + 1]
        no_biases = size[i + 1]
        no_params += (no_weights + no_biases)
    m_weights = weight_variable([no_params], init_weights)
    if init_variances is None:
        v_weights = small_variable([no_params])
    else:
        v_weights = tf.Variable(tf.constant(init_variances, dtype=tf.float32))
    return no_params, m_weights, v_weights, size


class Reg_NN(object):
    def __init__(self, input_size, hidden_size, output_size, training_size, y_mu, y_sigma):
        # input and output placeholders
        self.x = tf.placeholder(tf.float32, [None, input_size], name='x')
        self.y = tf.placeholder(tf.float32, [None, output_size], name='y')

        self.y_mu = tf.constant(y_mu, dtype=tf.float32, name='y_mu')
        self.y_sigma = tf.constant(y_sigma, dtype=tf.float32, name='y_sigma')

    def assign_optimizer(self, learning_rate=0.001):
        with tf.name_scope('optimiser'):
            self.train_step = tf.train.AdamOptimizer(learning_rate).minimize(self.cost)

    def assign_session(self):
        # Initializing the variables
        with tf.name_scope('initialisation'):
            init = tf.global_variables_initializer()
            init2 = tf.local_variables_initializer()
            config = tf.ConfigProto()
            config.gpu_options.allow_growth=True

            # launch a session
            self.sess = tf.Session(config=config)
            self.sess.run([init, init2])

    def train(self, x_train, y_train, no_epochs=100, batch_size=100, display_epoch=5):
        N = x_train.shape[0]
        if batch_size > N:
            batch_size = N

        sess = self.sess
        costs = []
        # Training cycle
        for epoch in range(no_epochs):
            perm_inds = list(range(x_train.shape[0]))
            np.random.shuffle(perm_inds)
            cur_x_train = x_train[perm_inds]
            cur_y_train = y_train[perm_inds]

            avg_cost = 0.
            total_batch = int(np.ceil(N * 1.0 / batch_size))
            # Loop over all batches
            for i in range(total_batch):
                start_ind = i * batch_size
                end_ind = np.min([(i + 1) * batch_size, N])
                batch_x = cur_x_train[start_ind:end_ind, :]
                batch_y = cur_y_train[start_ind:end_ind, :]
                # Run optimization op (backprop) and cost op (to get loss value)
                _, c = sess.run(
                    [self.train_step, self.cost],
                    feed_dict={self.x: batch_x, self.y: batch_y})
                # Compute average loss
                avg_cost += c / total_batch
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
        rmse = 0.
        total_batch = int(np.ceil(N * 1.0 / batch_size))
        # Loop over all batches
        for i in range(total_batch):
            start_ind = i * batch_size
            end_ind = np.min([(i + 1) * batch_size, N])
            batch_x = cur_x_test[start_ind:end_ind, :]
            batch_y = cur_y_test[start_ind:end_ind, :]
            # Run optimization op (backprop) and cost op (to get loss value)
            _ll, _rmse = sess.run(
                [self.test_loglik, self.rmse],
                feed_dict={self.x: batch_x, self.y: batch_y})

            # Compute average loss
            fraction = (end_ind - start_ind) / N
            ll += _ll * fraction
            rmse += _rmse * fraction

        return ll, rmse

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
        return s


""" Neural Network MLP Model """

""" Bayes MLP model using TFP"""


class BaysMLPRegressionTFP(Reg_NN):
    def __init__(self, input_size, hidden_size, output_size, training_size, y_mu, y_sigma,
                 no_train_samples=10, no_pred_samples=100, prev_means=None, prev_log_variances=None,
                 learning_rate=0.001,
                 prior_mean=0., prior_var=1.):

        super(BaysMLPRegressionTFP, self).__init__(input_size, hidden_size, output_size, training_size, y_mu, y_sigma)

        self.prior_mean = prior_mean
        self.prior_var = prior_var

        self.noise_var = tf.Variable(initial_value=-6., name='log_noise_variance')

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

    def _rmse(self, preds, targets):
        with tf.name_scope('rmse'):
            rmse = tf.sqrt(tf.reduce_mean(tf.squared_difference(preds, targets)))
            return rmse

    def _loglik(self, pred, targets):
        with tf.name_scope('loglik'):
            log_probs = distributions.Normal(targets, tf.exp(0.5 * self.noise_var)).log_prob(pred)
            return tf.reduce_mean(log_probs)

    def _test_loglik(self, pred, targets):
        with tf.name_scope('test_loglik'):
            log_probs = distributions.Normal(loc=targets, scale=tf.exp(0.5 * self.noise_var)).log_prob(pred)
            log_probs = tf.reduce_logsumexp(log_probs, axis=0) - tf.log(tf.to_float(tf.shape(log_probs)[0]))
            return tf.reduce_mean(log_probs)

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

        self.loglik = self._loglik(self.pred_train, self.y)
        self.KL = tf.div(sum(self.model.losses), self.training_size)

        self.cost = -self.loglik + self.KL

        # def test predictions and testing metrics
        self.pred_test = self._prediction(self.x, self.no_pred_samples)

        self.pred_test_actual = self.pred_test * self.y_sigma + self.y_mu
        self.targ_test_actual = self.y * self.y_sigma + self.y_mu

        self.rmse = self._rmse(self.pred_test_actual, self.targ_test_actual)
        self.test_loglik = self._test_loglik(self.pred_test, self.y)

    def make_metrics(self):
        with tf.name_scope('performance'):
            self.train_cost    = tf.placeholder(tf.float32, shape=None, name='train_cost_summary')
            self.train_logloss = tf.placeholder(tf.float32, shape=None, name='train_logloss_summary')
            self.train_kl      = tf.placeholder(tf.float32, shape=None, name='train_kl_summary')
            self.test_logloss  = tf.placeholder(tf.float32, shape=None, name='test_logloss_summary')
            self.test_rmse     = tf.placeholder(tf.float32, shape=None, name='test_rmse_summary')

            train_cost_summary    = tf.summary.scalar('train cost',     self.train_cost)
            train_logloss_summary = tf.summary.scalar('train logloss',  self.train_logloss)
            train_kl_summary      = tf.summary.scalar('train kl',       self.train_kl)
            test_logloss_summary  = tf.summary.scalar('test logloss',   self.test_logloss)
            test_rmse_summary     = tf.summary.scalar('test rmse',      self.test_rmse)
            noise_var_summary     = tf.summary.scalar('homoskedastic noise', tf.exp(0.5*self.noise_var))

            self.performance_metrics = tf.summary.merge_all()

    def log_metrics(self, train_cost, train_logloss, train_kl, test_logloss, test_rmse):
        return self.sess.run(self.performance_metrics, feed_dict={
            self.train_cost:train_cost,
            self.train_logloss: train_logloss,
            self.train_kl: train_kl,
            self.test_logloss: test_logloss,
            self.test_rmse: test_rmse
        })


""" Bayesian MLP Model """


class BayesMLPRegression(Reg_NN):
    def __init__(self, input_size, hidden_size, output_size, training_size, y_mu, y_sigma,
                 no_train_samples=10, no_pred_samples=100, prev_means=None, prev_log_variances=None,
                 learning_rate=0.001,
                 prior_mean=0., prior_var=1.):

        super(BayesMLPRegression, self).__init__(input_size, hidden_size, output_size, training_size, y_mu, y_sigma)

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

        self.prior_mean = prior_mean
        self.prior_var = prior_var

        self.noise_var = tf.Variable(initial_value=-6., name='log_noise_variance')

        self.no_layers = len(self.size)-1
        self.no_train_samples = no_train_samples
        self.no_pred_samples = no_pred_samples
        self.training_size = training_size

        # def train predictions and training metric production
        self.pred_train = self._prediction(self.x, self.no_train_samples)

        self.pred_train_actual = self.pred_train * self.y_sigma + self.y_mu
        self.targ_train_actual = self.y * self.y_sigma + self.y_mu

        self.loglik = self._loglik(self.pred_train, self.y)
        self.KL = tf.div(self._KL_term(), self.training_size)

        self.cost = -self.loglik + self.KL

        # def test predictions and testing metrics
        self.pred_test  = self._prediction(self.x, self.no_pred_samples)

        self.pred_test_actual = self.pred_test * self.y_sigma + self.y_mu
        self.targ_test_actual = self.y * self.y_sigma + self.y_mu

        self.rmse = self._rmse(self.pred_test_actual, self.targ_test_actual)
        self.test_loglik = self._test_loglik(self.pred_test, self.y)

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
        # for i in range(self.no_layers - 1):
        #     din = self.size[i]
        #     dout = self.size[i + 1]
        #     eps_w = tf.random_normal((K, din, dout), 0, 1, dtype=tf.float32)
        #     eps_b = tf.random_normal((K, 1, dout), 0, 1, dtype=tf.float32)
        #
        #     weights = tf.add(tf.multiply(eps_w, tf.exp(0.5 * self.W_v[i])), self.W_m[i])
        #     biases = tf.add(tf.multiply(eps_b, tf.exp(0.5 * self.b_v[i])), self.b_m[i])
        #     pre = tf.add(tf.einsum('mni,mio->mno', act, weights), biases)
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

    def _rmse(self, preds, targets):
        with tf.name_scope('rmse'):
            # pred_actual = pred_mean * self.y_sigma + self.y_mu
            # targets_actual = targets * self.y_sigma + self.y_mu

            rmse = tf.sqrt(tf.reduce_mean(tf.squared_difference(preds, targets)))
            return rmse

    # computes log likelihood of input for against target
    def _loglik(self, pred, targets):
        with tf.name_scope('loglik'):
            # se = tf.squared_difference(pred, targets)
            # const_term = - 0.5 * tf.log(tf.constant(2 * np.pi))
            # noise_term = - 0.5 * self.noise_var
            # se_norm_term = - tf.reduce_mean(se/(2* tf.exp(self.noise_var)))
            # return noise_term + se_norm_term + const_term
            log_probs = distributions.Normal(targets, tf.exp(0.5 * self.noise_var)).log_prob(pred)
            return tf.reduce_mean(log_probs)

    def _test_loglik(self, pred, targets):
        '''
        Calculate per data point
        note - does not un transform data before calculating - maybe it should?
        :param pred: input predictions
        :param targets: input targets
        :return: average test time log likelihood for the inputs - E[p(yi | xi, w)] = 1/N * SUM(i, N)[log( 1/M *SUM(j, M)[p(yi | xi, wj)] )]
        '''
        with tf.name_scope('test_loglik'):
            # se = tf.squared_difference(pred, targets)
            # probs = tf.div(1., tf.sqrt(tf.constant(2 * np.pi) * tf.exp(self.noise_var))) * tf.exp(-tf.div(se, 2 * tf.exp(self.noise_var)))
            # probs = tf.reduce_mean(probs, axis=0)
            # log_probs = tf.log(probs)
            # return tf.reduce_mean(log_probs)
            log_probs = distributions.Normal(loc=targets, scale=tf.exp(0.5 * self.noise_var)).log_prob(pred)
            log_probs = tf.reduce_logsumexp(log_probs, axis=0) - tf.log(tf.to_float(tf.shape(log_probs)[0]))
            return tf.reduce_mean(log_probs)


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

                    tf.summary.histogram("weights_mean", Wi_m)
                    tf.summary.histogram("bias_mean", bi_m)
                    tf.summary.histogram("weights_variance", Wi_v)
                    tf.summary.histogram("bais_variance", bi_v)

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

    def make_metrics(self):
        with tf.name_scope('performance'):
            self.train_cost    = tf.placeholder(tf.float32, shape=None, name='train_cost_summary')
            self.train_logloss = tf.placeholder(tf.float32, shape=None, name='train_logloss_summary')
            self.train_kl      = tf.placeholder(tf.float32, shape=None, name='train_kl_summary')
            self.test_logloss  = tf.placeholder(tf.float32, shape=None, name='test_logloss_summary')
            self.test_rmse     = tf.placeholder(tf.float32, shape=None, name='test_rmse_summary')

            train_cost_summary    = tf.summary.scalar('train cost',     self.train_cost)
            train_logloss_summary = tf.summary.scalar('train logloss',  self.train_logloss)
            train_kl_summary      = tf.summary.scalar('train kl',       self.train_kl)
            test_logloss_summary  = tf.summary.scalar('test logloss',   self.test_logloss)
            test_rmse_summary     = tf.summary.scalar('test rmse',      self.test_rmse)
            noise_var_summary     = tf.summary.scalar('homoskedastic noise', tf.exp(0.5*self.noise_var))

            self.performance_metrics = tf.summary.merge_all()

    def log_metrics(self, train_cost, train_logloss, train_kl, test_logloss, test_rmse):
        return self.sess.run(self.performance_metrics, feed_dict={
            self.train_cost:train_cost,
            self.train_logloss: train_logloss,
            self.train_kl: train_kl,
            self.test_logloss: test_logloss,
            self.test_rmse: test_rmse
        })


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

        self.prior_mean = prior_mean
        self.prior_var = prior_var

        self.noise_var = tf.Variable(initial_value=-6., name='log_noise_variance')

        self.no_layers = len(self.size)-1
        self.no_train_samples = no_train_samples
        self.no_pred_samples = no_pred_samples
        self.training_size = training_size
        self.skips = skips

        # def train predictions and training metric production
        self.pred_train = self._prediction(self.x, self.no_train_samples)

        self.pred_train_actual = self.pred_train * self.y_sigma + self.y_mu
        self.targ_train_actual = self.y * self.y_sigma + self.y_mu

        self.loglik = self._loglik(self.pred_train, self.y)
        self.KL = tf.div(self._KL_term(), self.training_size)

        self.cost = -self.loglik + self.KL

        # def test predictions and testing metrics
        self.pred_test  = self._prediction(self.x, self.no_pred_samples)

        self.pred_test_actual = self.pred_test * self.y_sigma + self.y_mu
        self.targ_test_actual = self.y * self.y_sigma + self.y_mu

        self.rmse = self._rmse(self.pred_test_actual, self.targ_test_actual)
        self.test_loglik = self._test_loglik(self.pred_test, self.y)

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
        # for i in range(self.no_layers - 1):
        #     din = self.size[i]
        #     dout = self.size[i + 1]
        #     eps_w = tf.random_normal((K, din, dout), 0, 1, dtype=tf.float32)
        #     eps_b = tf.random_normal((K, 1, dout), 0, 1, dtype=tf.float32)
        #
        #     weights = tf.add(tf.multiply(eps_w, tf.exp(0.5 * self.W_v[i])), self.W_m[i])
        #     biases = tf.add(tf.multiply(eps_b, tf.exp(0.5 * self.b_v[i])), self.b_m[i])
        #     pre = tf.add(tf.einsum('mni,mio->mno', act, weights), biases)
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

    def _rmse(self, preds, targets):
        with tf.name_scope('rmse'):
            # pred_actual = pred_mean * self.y_sigma + self.y_mu
            # targets_actual = targets * self.y_sigma + self.y_mu

            rmse = tf.sqrt(tf.reduce_mean(tf.squared_difference(preds, targets)))
            return rmse

    # computes log likelihood of input for against target
    def _loglik(self, pred, targets):
        with tf.name_scope('loglik'):
            # se = tf.squared_difference(pred, targets)
            # const_term = - 0.5 * tf.log(tf.constant(2 * np.pi))
            # noise_term = - 0.5 * self.noise_var
            # se_norm_term = - tf.reduce_mean(se/(2* tf.exp(self.noise_var)))
            # return noise_term + se_norm_term + const_term
            log_probs = distributions.Normal(targets, tf.exp(0.5 * self.noise_var)).log_prob(pred)
            return tf.reduce_mean(log_probs)

    def _test_loglik(self, pred, targets):
        '''
        Calculate per data point
        note - does not un transform data before calculating - maybe it should?
        :param pred: input predictions
        :param targets: input targets
        :return: average test time log likelihood for the inputs - E[p(yi | xi, w)] = 1/N * SUM(i, N)[log( 1/M *SUM(j, M)[p(yi | xi, wj)] )]
        '''
        with tf.name_scope('test_loglik'):
            # se = tf.squared_difference(pred, targets)
            # probs = tf.div(1., tf.sqrt(tf.constant(2 * np.pi) * tf.exp(self.noise_var))) * tf.exp(-tf.div(se, 2 * tf.exp(self.noise_var)))
            # probs = tf.reduce_mean(probs, axis=0)
            # log_probs = tf.log(probs)
            # return tf.reduce_mean(log_probs)
            log_probs = distributions.Normal(loc=targets, scale=tf.exp(0.5 * self.noise_var)).log_prob(pred)
            log_probs = tf.reduce_logsumexp(log_probs, axis=0) - tf.log(tf.to_float(tf.shape(log_probs)[0]))
            return tf.reduce_mean(log_probs)


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

    def make_metrics(self):
        with tf.name_scope('performance'):
            self.train_cost    = tf.placeholder(tf.float32, shape=None, name='train_cost_summary')
            self.train_logloss = tf.placeholder(tf.float32, shape=None, name='train_logloss_summary')
            self.train_kl      = tf.placeholder(tf.float32, shape=None, name='train_kl_summary')
            self.test_logloss  = tf.placeholder(tf.float32, shape=None, name='test_logloss_summary')
            self.test_rmse     = tf.placeholder(tf.float32, shape=None, name='test_rmse_summary')

            train_cost_summary    = tf.summary.scalar('train cost',     self.train_cost)
            train_logloss_summary = tf.summary.scalar('train logloss',  self.train_logloss)
            train_kl_summary      = tf.summary.scalar('train kl',       self.train_kl)
            test_logloss_summary  = tf.summary.scalar('test logloss',   self.test_logloss)
            test_rmse_summary     = tf.summary.scalar('test rmse',      self.test_rmse)
            noise_var_summary     = tf.summary.scalar('homoskedastic noise', tf.exp(0.5*self.noise_var))

            self.performance_metrics = tf.summary.merge_all()

    def log_metrics(self, train_cost, train_logloss, train_kl, test_logloss, test_rmse):
        return self.sess.run(self.performance_metrics, feed_dict={
            self.train_cost:train_cost,
            self.train_logloss: train_logloss,
            self.train_kl: train_kl,
            self.test_logloss: test_logloss,
            self.test_rmse: test_rmse
        })

