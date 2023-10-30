import numpy as np
import numpy.linalg
import os
import torch

# # a = np.array([5., 10.])
# a = np.array([20., 30., 40., 50.])
# # print(np.exp(a))

# # softmax
# a = a * 0.05
# b = np.exp(-a) / np.sum(np.exp(-a))
# print(b)
# print(np.sum(b))

# print(40 / np.sqrt(2))

# # f = M / (2 * tan (fov / 2))
# # fov = 2*arctan(pixelNumber/(2*focalLength)) * (180/pi)(*)
# D2R = np.pi / 180
# f_x = 1920 / (2 * np.tan((110 * D2R) / 2))
# f_y = 1080 / (2 * np.tan((80 * D2R) / 2))
# print(f_x, f_y)

a = np.array([[4, 9],
              [9, 4]])

b = np.sqrt(a)
print(b)
b = b * 3
print(b)
