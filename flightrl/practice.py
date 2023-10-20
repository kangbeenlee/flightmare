import numpy as np
import numpy.linalg
import os
import torch

a = np.array([10., 10.5, 12.])
# print(np.exp(a))

# softmax
a = a * 1.5
b = np.exp(-a) / np.sum(np.exp(-a))
print(b)
print(np.sum(b))