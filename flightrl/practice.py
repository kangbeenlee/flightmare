import numpy as np
import numpy.linalg
import os
import torch

# a = np.array([13., 13., 13.])
# # print(np.exp(a))

# # softmax
# a = a * 0.6
# print(np.exp( - 0.6 * 13))
# b = np.exp(-a) / np.sum(np.exp(-a))
# print(b)
# print(np.sum(b))

# lst = [12.617333354206455, 1.2498261503308377, 1.1182362501309542]
lst = [12.6, 1.2, 1.1]
idx = np.argmin(lst)
print(idx)