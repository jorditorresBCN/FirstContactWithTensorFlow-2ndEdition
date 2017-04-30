import tensorflow as tf
tf.InteractiveSession()
a = tf.zeros((2,2))
b = tf.ones((2,2))
print a.eval()
print b .eval()
print tf.reduce_sum(b, reduction_indices=1).eval()
print a.get_shape()
print tf.reshape(a, (1, 4)).eval()
