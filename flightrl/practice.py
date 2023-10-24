import numpy as np
import numpy.linalg
import os
import torch

a = np.array([7., 8., 12., 15., 20.])
# print(np.exp(a))

# softmax
a = a * 0.3
b = np.exp(-a) / np.sum(np.exp(-a))
print(b)
print(np.sum(b))

print(np.sqrt(3) * 5) # 8.6