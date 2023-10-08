import numpy as np
import torch
import matplotlib.pyplot as plt


x1 = np.linspace(0, 4, 100)
# x2 = np.linspace(1.5, 10, 100)
# y1 = np.exp(-10. * x1 ** 3)
y1 = np.exp(-0.1 * x1 ** (5))
print(np.exp(-0.1 * (0.8 ** 5)))
print(np.exp(-0.1 * (1.2 ** 5)))
print(np.exp(-0.1 * (2.0 ** 5)))

plt.plot(x1, y1)
# plt.plot(x2, y2)
plt.axis('equal')
plt.show()

# import numpy as np
# import matplotlib.pylab as plt
  
# def func(x):
#     return x / np.sqrt(1 + x**2) 
 
# x = np.arange(0, 5, 0.01)
# plt.plot(x, func(x), linestyle='-')
# plt.ylim(-10, 10)
# plt.axis('equal')
# plt.show() 