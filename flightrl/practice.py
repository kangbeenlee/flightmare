import numpy as np
import numpy.linalg

a = np.array([1., 1.1, 12.])
# print(np.exp(a))

# softmax
a = a * 0.6
b = np.exp(-a) / np.sum(np.exp(-a))
print(b)
print(np.sum(b))