# import numpy as np
# import numpy.linalg
# import os
# import torch

# a = np.array([3., 4.2, 6.5])
# # print(np.exp(a))

# # softmax
# a = a * 0.6
# b = np.exp(-a) / np.sum(np.exp(-a))
# print(b)
# print(np.sum(b))


# print("best {}".format(None))


lst = [1,2,3,4,5]
idx, max = max(enumerate(lst), key=lambda x: x[1])
print(idx, max)