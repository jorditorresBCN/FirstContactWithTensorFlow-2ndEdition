import numpy as np
import tensorflow as tf


tf.set_random_seed(1234)

with tf.name_scope('data'):
    with tf.name_scope('x'):
        x = tf.random_normal([100], mean=0.0, stddev=0.9, name='rand_normal_x')
    with tf.name_scope('y'):
        y_true = x * tf.constant(0.1, name='real_slope') + tf.constant(0.3, name='bias') + tf.random_normal([100], mean=0.0, stddev=0.05, name='rand_normal_y')

with tf.name_scope('W'):
    W = tf.Variable(tf.random_uniform([], minval=-1.0, maxval=1.0))
    tf.summary.scalar('function/W', W)

with tf.name_scope('b'):
    b = tf.Variable(tf.zeros([]))
    tf.summary.scalar('function/b', b)

with tf.name_scope('function'):
    y_pred = W * x + b
        

with tf.name_scope('error'):
    loss = tf.reduce_mean(tf.square(y_pred - y_true))
    tf.summary.scalar('error', loss)

with tf.name_scope('train'):
    optimizer = tf.train.GradientDescentOptimizer(0.05)
    train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

sess = tf.Session()

merged = tf.summary.merge_all()
writer = tf.summary.FileWriter('/tmp/regression/run1', sess.graph)

sess.run(init)

for step in range(1, 101):
    _, summary_str, slope, intercept, error = sess.run([train, merged, W, b, loss])
    if step % 10 == 0:
        writer.add_summary(summary_str, step)
        print('Step %.3d: W = %.5f; b = %.5f; loss = %.5f' % (step, slope, intercept, error))