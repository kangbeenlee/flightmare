import numpy as np
import torch
import matplotlib.pyplot as plt



def func(x):
    # return 0.9*(1 - np.exp(-0.00001 * (x ** 4)))
    return np.exp(-0.01 * (x ** 3))

# Scalar cov_reward = exp(-0.1 * pow(avg_position_cov_norm, 5));
# Scalar target_cov_reward = exp(-0.01 * pow(target_cov_norm, 3));
# exp(-0.01 * pow(target_cov_norm, 3));
# exp(-10.0 * pow(theta, 3));

x1 = np.linspace(0, 30, 100)

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
# plt.plot(x1, 0.9*np.ones_like(x1), linestyle='dashed')
# plt.axis('equal')
plt.show()