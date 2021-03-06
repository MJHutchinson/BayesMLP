import numpy as np
import tensorflow as tf
import tensorflow_probability.python.distributions as d

def gaussian_init(mean, sigma, shape):
    return mean + sigma * np.random.randn(*shape).astype(np.float32)

class Paramter(object):
    def __init__(self, value, prior, variables=None):
        self.value = value
        self.prior = prior
        self.varables = variables

    def _KL(self):
        """
                compute KL(value||prior) for each weight
                assume:
                    (1) value is a diagonal gaussian
                    (2) prior is a diagonal gaussian or an inverse-gamme
                    (3) if prior is an inverse-gamma, it is a hyper-prior
                """
        if isinstance(self.value, d.Normal):
            if isinstance(self.prior, d.Normal):
                return d.kl_divergence(self.value, self.prior)

            elif isinstance(self.prior, d.InverseGamma):
                m = tf.to_float(tf.reduce_prod(tf.shape(self.value.loc)))
                S = tf.reduce_sum(self.value.scale ** 2 + self.value.loc ** 2)
                m_plus_2alpha_plus_2 = m + 2.0 * self.prior.concentration + 2.0
                S_plus_2beta = S + 2.0 * self.prior.rate
                s_star = S_plus_2beta / m_plus_2alpha_plus_2

                tf.summary.scalar(name='prior_std', tensor=s_star)

                return d.kl_divergence(self.value, d.Normal(0, tf.sqrt(s_star)))

    def KL(self):
        """
        Computes the total KL for the given
        :return:
        """
        return tf.reduce_sum(self._KL())

    def pruning_from_KL(self):
        """
        Quantifies how much each unit has been pruned by computing the KL of the weights of a given neuron. KL
        normalised to be per weight.
        ONLY works on weight parameters at the moment - does not incorporate biases and will not calculate correctly
        on biases
        :return: a list confining KL per neuron
        """
        return tf.reduce_mean(self._KL(), axis=0)

    def pruning_from_SNR(self):
        """
        Quantifies how much each weight has been pruned - per weight NOT per neuron. From Bayes by Backprop
        :return:
        """
        ratio = tf.divide(tf.abs(self.value.loc), tf.abs(self.value.scale))
        return tf.reshape(ratio, [-1])


def make_weight_parameter(shape, prior_var=1., hyper_prior=False, name=None):
    s2 = prior_var # TODO: scale with number of hidden units?, Potentially needs fudge factor if so for bias from FVB:DVI
    sigma = np.ones(shape) * np.sqrt(s2)
    log_sigma = np.ones(shape) * -5.5 # np.log(sigma) # TODO: would usually initialise this to be around 10e-11?
    log_sigma = tf.Variable(log_sigma, dtype=tf.float32, name='log_sigma')
    sigma = tf.exp(log_sigma)

    if hyper_prior:
        a = 4.4798
        alpha = tf.Variable(a, dtype=tf.float32, trainable=False)
        beta  = tf.Variable((1+a) * s2, dtype=tf.float32, trainable=False)

        mean = tf.truncated_normal(shape, stddev=0.1)#np.sqrt(s2)) # TODO: consider how to initialise; usually with small var around 0.1, not the regular std
        mean = tf.Variable(mean, name='mean')
        value = d.Normal(mean, sigma, name='value')
        prior = d.InverseGamma(alpha, beta, name='prior')

        return Paramter(value, prior, {'mean': mean, 'log_sigma': log_sigma})
    else:
        mean = tf.truncated_normal(shape, stddev=0.1)#np.sqrt(s2)) # TODO: consider how to initialise; usually with small var around 0.1, not the regular std
        mean = tf.Variable(mean, name='mean')
        value = d.Normal(mean, sigma, name='value')

        prior_mean = tf.Variable(np.broadcast_to(0.0, shape), dtype=tf.float32, trainable=False)
        prior_var  = tf.Variable(np.broadcast_to(s2, shape), dtype=tf.float32, trainable=False)
        prior = d.Normal(prior_mean, tf.sqrt(prior_var), name='prior')

        return Paramter(value, prior, {'mean': mean, 'log_sigma': log_sigma})

