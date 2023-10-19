import numpy as np
import torch
import matplotlib.pyplot as plt



def func(x):
    return 1 - np.exp(-0.005 * (x ** 2))

# Scalar cov_reward = exp(-0.1 * pow(avg_position_cov_norm, 5));

x1 = np.linspace(0, 30, 100)
# x2 = np.linspace(1.5, 10, 100)
# y1 = np.exp(-10. * x1 ** 3)

# alpha = -0.01
# beta = 5

# y1 = np.exp(alpha * x1 ** (beta))
# print(np.exp(alpha * (1.0 ** beta)))
# print(np.exp(alpha * (2.0 ** beta)))
# print(np.exp(alpha * (5.0 ** beta)))
# print(np.exp(alpha * (10.0 ** beta)))
# print(np.exp(alpha * (20.0 ** beta)))

y1 = func(x1)
print(func(1.0))
print(func(2.0))
print(func(5.0))
print(func(10.0))
print(func(20.0))
print(func(30.0))

plt.plot(x1, y1)
# plt.plot(x2, y2)
# plt.axis('equal')
plt.show()