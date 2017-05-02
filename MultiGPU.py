# Multi GPU Basic example
# code source: Github (2016) Aymeric Damien: https://github.com/aymericdamien/TensorFlow-Examples 
'''
This tutorial requires your machine to have 2 GPUs
"/cpu:0": The CPU of your machine.
"/gpu:0": The first GPU of your machine
"/gpu:1": The second GPU of your machine
'''

import tensorflow as tf
import datetime

# Processing Units logs
log_device_placement = True

# num of multiplications to perform
n = 10

# shape of the matrix
matrix_shape = [10000, 10000]


def matpow(M, n):
    if n < 1:  # Abstract cases where n < 1
        return M
    else:
        return tf.matmul(M, matpow(M, n-1))

'''
Example: compute A^n + B^n on 2 GPUs
Results on 16 cores with 2 NVIDIA K80:
 * Only CPU computation time: 0:02:00.965574
 * Single GPU computation time: 0:00:24.933976
 * Multi GPU (x2) computation time: 0:00:08.771551
'''

'''
Only CPU computing
'''
with tf.device('/cpu:0'):
    # Creates two random matrix with shape (1e4, 1e4)
    a = tf.random_normal(matrix_shape)
    b = tf.random_normal(matrix_shape)
    # Compute A^n and B^n and store in a tensor
    r01 = matpow(a, n)
    r02 = matpow(b, n)
    sum = r01 + r02  # Addition of elements:  A^n + B^n

t1_0 = datetime.datetime.now()
with tf.Session(config=tf.ConfigProto(log_device_placement=log_device_placement)) as sess:
    # Runs the op.
    sess.run(sum)
t2_0 = datetime.datetime.now()

# Clean the graph to start another computation
tf.reset_default_graph()

'''
Single GPU computing
'''
with tf.device('/gpu:0'):
    # Creates two random matrix with shape (1e4, 1e4)
    a = tf.random_normal(matrix_shape)
    b = tf.random_normal(matrix_shape)
    # Compute A^n and B^n and store in a tensor
    r01 = matpow(a, n)
    r02 = matpow(b, n)

with tf.device('/cpu:0'):
    sum = r01 + r02  # Addition of elements:  A^n + B^n

t1_1 = datetime.datetime.now()
with tf.Session(config=tf.ConfigProto(log_device_placement=log_device_placement)) as sess:
    # Runs the op.
    sess.run(sum)
t2_1 = datetime.datetime.now()

# Clean the graph to start another computation
tf.reset_default_graph()

'''
Multi GPU computing
'''
# GPU:0 computes A^n
with tf.device('/gpu:0'):
    # Create one random matrix with shape (1e4, 1e4)
    a = tf.random_normal(matrix_shape)
    # Compute A^n and store result in a tensor
    r11 = matpow(a, n)

# GPU:1 computes B^n
with tf.device('/gpu:1'):
    # Create one random matrix with shape (1e4, 1e4)
    b = tf.random_normal(matrix_shape)
    # Compute b^n and store result in a tensor
    r12 = matpow(b, n)

with tf.device('/cpu:0'):
    sum = r11 + r12  # Addition of elements:  A^n + B^n

t1_2 = datetime.datetime.now()
with tf.Session(config=tf.ConfigProto(log_device_placement=log_device_placement)) as sess:
    # Runs the op.
    sess.run(sum)
t2_2 = datetime.datetime.now()

print("Only CPU computation time: " + str(t2_0 - t1_0))
print("Single GPU computation time: " + str(t2_1 - t1_1))
print("Multi GPU computation time: " + str(t2_2 - t1_2))
