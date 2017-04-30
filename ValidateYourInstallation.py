# to remove warnings saying if you build TensorFlow from source it can be faster on your machine
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf

#test
import tensorflow as tf
hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
print(sess.run(hello))
 
