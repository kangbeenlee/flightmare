import numpy as np
import numpy.linalg
import os
import torch

a = np.array([3., 4.2, 6.5])
# print(np.exp(a))

# softmax
a = a * 0.6
b = np.exp(-a) / np.sum(np.exp(-a))
print(b)
print(np.sum(b))