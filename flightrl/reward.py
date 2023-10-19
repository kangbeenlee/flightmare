import numpy as np
import torch
import matplotlib.pyplot as plt


# Scalar cov_reward = exp(-0.1 * pow(avg_position_cov_norm, 5));

x1 = np.linspace(0, 10, 100)
# x2 = np.linspace(1.5, 10, 100)
# y1 = np.exp(-10. * x1 ** 3)

alpha = -0.01
beta = 3
y1 = np.exp(alpha * x1 ** (beta))
print(np.exp(alpha * (0.8 ** beta)))
print(np.exp(alpha * (1.2 ** beta)))
print(np.exp(alpha * (2.0 ** beta)))
print(np.exp(alpha * (5.0 ** beta)))
print(np.exp(alpha * (10.0 ** beta)))
print(np.exp(alpha * (15.0 ** beta)))
print(np.exp(alpha * (50.0 ** beta)))

plt.plot(x1, y1)
# plt.plot(x2, y2)
plt.axis('equal')
plt.show()