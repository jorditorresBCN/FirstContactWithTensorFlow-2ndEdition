import numpy as np
a = np.zeros((2,2))
b = np.ones((2,2))
print "a= ", a
print "b= ", b
print "np.sum(b, axis=1)= " , np.sum(b, axis=1)
print "a.shape= ", a.shape
print "np.reshape(a, (1,4))= ", np.reshape(a, (1,4))
