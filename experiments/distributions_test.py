import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow_probability.python.distributions as D

target_mean = tf.Variable([3.], trainable=False)
target_var = tf.Variable([3.], trainable=False)
target_dist = D.Normal(target_mean, target_var)

mean = tf.Variable([0.], trainable=True)
var = tf.Variable([1.], trainable=True)
dist = D.Normal(mean, var)

cost = D.kl_divergence(dist, target_dist)

optim_step = tf.train.AdamOptimizer().minimize(cost)

sess = tf.Session()

init = tf.global_variables_initializer()
sess.run([init])

for epoch in range(10000):
    _, c, m, v = sess.run([optim_step, cost, mean, var])
    print(c, m, v)

