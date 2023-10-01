import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Create a figure and a 3D axis
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Create data
x = np.linspace(0, 4 * np.pi, 100)
y = np.sin(x)
z = np.cos(x)

line, = ax.plot(x, y, z, '-b')

# Update function for animation
def update(num, x, y, z, line):
    # Rotate the sine wave in y and z axis
    y = np.sin(x + num * 0.1)
    z = np.cos(x + num * 0.1)
    
    line.set_data(x, y)
    line.set_3d_properties(z)
    return line,

# Set up the animation
ani = FuncAnimation(fig, update, frames=100, fargs=[x, y, z, line], interval=50)

# Set axis limits
ax.set_xlim([0, 4*np.pi])
ax.set_ylim([-1.5, 1.5])
ax.set_zlim([-1.5, 1.5])

# Display the animation
plt.show()