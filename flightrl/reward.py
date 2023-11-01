import numpy as np
import torch
import matplotlib.pyplot as plt



def func(x):
    return np.exp(-0.1 * (x ** 5))
    # return np.exp(-0.001 * (x ** 4))

# Scalar cov_reward = exp(-0.1 * pow(avg_position_cov_norm, 5));
# Scalar target_cov_reward = exp(-0.01 * pow(target_cov_norm, 3));
# exp(-0.01 * pow(target_cov_norm, 3));
# exp(-10.0 * pow(theta, 3));

x1 = np.linspace(0, 15, 100)
y1 = func(x1)
print(func(0.2))
print(func(0.6))
print(func(1.0))
print(func(2.0))
print(func(3.0))

plt.plot(x1, y1)
# plt.plot(x1, 0.9*np.ones_like(x1), linestyle='dashed')
# plt.axis('equal')
plt.show()