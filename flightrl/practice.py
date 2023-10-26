import numpy as np
import numpy.linalg
import os
import torch

# a = np.array([5., 10.])
a = np.array([1., 5., 10., 20., 30.])
# print(np.exp(a))

# softmax
a = a * 0.05
b = np.exp(-a) / np.sum(np.exp(-a))
print(b)
print(np.sum(b))