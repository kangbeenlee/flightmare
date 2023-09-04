import numpy as np
import torch
import matplotlib.pyplot as plt


# x1 = np.linspace(0, 8, 100)
# y1 = np.exp(-x1)
# y2 = 8 - x2

# plt.plot(x1, y1)
# plt.plot(x2, y2)
# plt.axis('equal')
# plt.show()

x1 = np.linspace(0, 1, 100)
x2 = np.linspace(1, 15, 100)
y1 = -np.ones(100)
y2 = -1/7 * x2 + 1

plt.plot(x1, y1)
plt.plot(x2, y2)
plt.axis('equal')
plt.show()