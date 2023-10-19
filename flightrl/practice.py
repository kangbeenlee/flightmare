import numpy as np
import numpy.linalg
import os
import torch

a = np.array([10., 13., 11.])
# print(np.exp(a))

# softmax
a = a * 0.6
print(np.exp( - 0.6 * 13))
b = np.exp(-a) / np.sum(np.exp(-a))
print(b)
print(np.sum(b))