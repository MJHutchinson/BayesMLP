import tensorflow as tf
import tensorflow_probability.python.distributions as distributions
import numpy as np


class tests:

    def __init__(self):

        self.pred = tf.placeholder(tf.float32, shape=[None, None], name='pred')
        self.targ = tf.placeholder(tf.float32, shape=[None, None], name='targ')

        self.noise_var = tf.placeholder(tf.float32, shape=[], name='noise_var')

        self.rmse = self._rmse(self.pred, self.targ)
        self.ll1 = self._loglik_1(self.pred, self.targ)
        self.ll2 = self._loglik_2(self.pred, self.targ)
        self.tll1 = self._test_loglik_1(self.pred, self.targ)
        self.tll2 = self._test_loglik_2(self.pred, self.targ)

        init = tf.global_variables_initializer()
        init2 = tf.local_variables_initializer()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        self.no_layers = 1
        self.W_m = [tf.constant([[2,1,1]], tf.float32)]
        self.W_v = [tf.log(tf.constant([[1,5,1]], tf.float32))]

        self.prior_W_m = [1.]
        self.prior_W_v = [1.]

        self.b_m = [tf.constant([[1,1,1]], tf.float32)]
        self.b_v = [tf.log(tf.constant([[1,1,1]], tf.float32))]

        self.prior_b_m = [1.]
        self.prior_b_v = [1.]

        self.kl1 = self._KL_1()
        self.kl2 = self._KL_2()

        # launch a session
        self.sess = tf.Session(config=config)
        self.sess.run([init, init2])

    def _rmse(self, preds, targets):
        with tf.name_scope('rmse'):
            rmse = tf.sqrt(tf.reduce_mean(tf.squared_difference(preds, targets)))
            return rmse

    # computes log likelihood of input for against target
    def _loglik_1(self, pred, targets):
        with tf.name_scope('loglik'):
            log_probs = distributions.Normal(targets, tf.exp(0.5 * self.noise_var)).log_prob(pred)
            return tf.reduce_mean(log_probs)

    # computes log likelihood of input for against target
    def _loglik_2(self, pred, targets):
        with tf.name_scope('loglik'):
            se = tf.squared_difference(pred, targets)
            const_term = - 0.5 * tf.log(tf.constant(2 * np.pi))
            noise_term = - 0.5 * self.noise_var
            se_norm_term = - tf.reduce_mean(se/(2* tf.exp(self.noise_var)))
            return noise_term + se_norm_term + const_term


    def _test_loglik_1(self, pred, targets):
        with tf.name_scope('test_loglik'):
            log_probs = distributions.Normal(loc=targets, scale=tf.exp(0.5 * self.noise_var)).log_prob(pred)
            log_probs = tf.reduce_logsumexp(log_probs, axis=0) - tf.log(tf.to_float(tf.shape(log_probs)[0]))
            return tf.reduce_mean(log_probs)


    def _test_loglik_2(self, pred, targets):
        with tf.name_scope('test_loglik'):
            se = tf.squared_difference(pred, targets)
            probs = tf.div(1., tf.sqrt(tf.constant(2 * np.pi) * tf.exp(self.noise_var))) * tf.exp(-tf.div(se, 2 * tf.exp(self.noise_var)))
            probs = tf.reduce_mean(probs, axis=0)
            log_probs = tf.log(probs)
            return tf.reduce_mean(log_probs)

    def _KL_1(self):
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


    def _KL_2(self):
        with tf.name_scope('kl'):
            kl = 0
            for i in range(self.no_layers):
                with tf.name_scope(f'layer_{i}'):

                    m, v = self.W_m[i], self.W_v[i]
                    m0, v0 = self.prior_W_m[i], self.prior_W_v[i]

                    dout = m.get_shape().as_list()[-1]
                    din = m.get_shape().as_list()[-2]

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

    def get(self, pred, targ, noise_var):
        return self.sess.run([self.rmse, self.ll1, self.ll2, self.tll1, self.tll2, self.kl1, self.kl2],
                                               feed_dict={self.pred: pred, self.targ: targ, self.noise_var: noise_var})


if __name__ == '__main__':
    test = tests()

    noise_var = -2
    pred = np.array([[1, 1, 1, 1],
                     [2, 2, 2, 2]])

    targ = np.array([[1, 2, 5, 0],
                     [2, 1, 4, 3]])

    print(test.get(pred, targ, noise_var))

    noise_var = -1
    pred = np.array([[1, 1, 1, 1],
                     [2, 2, 2, 2]])

    targ = np.array([[1, 1, 3, 3],
                     [2, 2, 2, 2]])

    print(test.get(pred, targ, noise_var))

    kl = 0

    m, v =   np.array([1,1,1]), np.log(np.array([1,1,1]))
    m0, v0 = np.array([1]), np.array([1])

    dout = m.shape[-1]
    din = 1 # m.shape[-2]

    const_term = -0.5 * dout * din
    log_std_diff = 0.5 * np.sum(np.log(v0) - v)
    mu_diff_term = 0.5 * np.sum((np.exp(v) + (m0 - m) ** 2) / v0)
    kl += const_term + log_std_diff + mu_diff_term

    m, v =   np.array([1,1,1]), np.log(np.array([1,1,1]))
    m0, v0 = np.array([1]), np.array([1])

    const_term = -0.5 * dout
    log_std_diff = 0.5 * np.sum(np.log(v0) - v)
    mu_diff_term = 0.5 * np.sum((np.exp(v) + (m0 - m) ** 2) / v0)
    kl += const_term + log_std_diff + mu_diff_term
    print(kl)