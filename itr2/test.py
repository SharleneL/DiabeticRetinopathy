__author__ = 'luoshalin'


import numpy as np

a = np.array([1, 2, 3])
b = np.array([2, 2, 3])
c = np.array([4, 2, 5])

l = [a, b, c]

c = np.r_[a[None,:],b[None,:]]
m = np.r_[k[None,:] for k in l]