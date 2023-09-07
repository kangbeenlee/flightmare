import numpy as np
import torch
import matplotlib.pyplot as plt


x1 = np.linspace(0, 1, 100)
x2 = np.linspace(1, 12, 100)
y1 = 10 * x1
y2 = - 10/7 * (x2 - 8)

plt.plot(x1, y1)
plt.plot(x2, y2)
plt.axis('equal')
plt.show()

# x1 = np.linspace(0, 1, 100)
# x2 = np.linspace(1, 15, 100)
# y1 = x1
# y2 = -(x2 - 8)/7

# plt.plot(x1, y1)
# plt.plot(x2, y2)
# plt.axis('equal')
# plt.show()

if not None:
    print("hello")