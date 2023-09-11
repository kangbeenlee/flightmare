import numpy as np
import torch
import matplotlib.pyplot as plt


x1 = np.linspace(0, 10, 100)
y1 = np.exp(-x1)

plt.plot(x1, y1)
plt.axis('equal')
plt.show()