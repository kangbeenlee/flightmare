import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg

# Load data from txt files
gt_x = np.array([float(value) for line in open("/home/kblee/catkin_ws/src/flightmare/flightlib/include/flightlib/data/tracking_output/gt_x.txt") for value in line.split()])
gt_y = np.array([float(value) for line in open("/home/kblee/catkin_ws/src/flightmare/flightlib/include/flightlib/data/tracking_output/gt_y.txt") for value in line.split()])
gt_z = np.array([float(value) for line in open("/home/kblee/catkin_ws/src/flightmare/flightlib/include/flightlib/data/tracking_output/gt_z.txt") for value in line.split()])

estim_x = np.array([float(value) for line in open("/home/kblee/catkin_ws/src/flightmare/flightlib/include/flightlib/data/tracking_output/estim_x.txt") for value in line.split()])
estim_y = np.array([float(value) for line in open("/home/kblee/catkin_ws/src/flightmare/flightlib/include/flightlib/data/tracking_output/estim_y.txt") for value in line.split()])
estim_z = np.array([float(value) for line in open("/home/kblee/catkin_ws/src/flightmare/flightlib/include/flightlib/data/tracking_output/estim_z.txt") for value in line.split()])

cov_x = np.array([float(value) for line in open("/home/kblee/catkin_ws/src/flightmare/flightlib/include/flightlib/data/tracking_output/cov_x.txt") for value in line.split()])
cov_y = np.array([float(value) for line in open("/home/kblee/catkin_ws/src/flightmare/flightlib/include/flightlib/data/tracking_output/cov_y.txt") for value in line.split()])
cov_z = np.array([float(value) for line in open("/home/kblee/catkin_ws/src/flightmare/flightlib/include/flightlib/data/tracking_output/cov_z.txt") for value in line.split()])

time = np.array([float(line.strip()) for line in open("/home/kblee/catkin_ws/src/flightmare/flightlib/include/flightlib/data/tracking_output/time.txt")])

print(gt_x[0], gt_y[0], gt_z[0])
print(estim_x[0], estim_y[0], estim_z[0])

def compute3DEllipsoidCoordinates(cx, cy, cz, x_axis, y_axis, z_axis):
    """ Compute ellipsoid coordinates

    Args:
        cx (float): center x of ellipsoid
        cy (float): center y of ellipsoid
        cz (float): center z of ellipsoid
        x_axis (float): x axis of ellipsoid
        y_axis (float): y axis of ellipsoid
        z_axis (float): z axis of ellipsoid

    Returns:
        xs (np.array): set of ellipsoid's x coordinates
        ys (np.array): set of ellipsoid's y coordinates
        zs (np.array): set of ellipsoid's z coordinates
    """
    P = np.array([[x_axis,0,0],
                  [0,y_axis,0],
                  [0,0,z_axis]])
    center = [cx, cy, cz]
    # Find the rotation matrix and radii of the axes
    _, radii, rotation = linalg.svd(P)

    # Calculate cartesian coordinates for the ellipsoid surface
    u = np.linspace(0.0, 2.0 * np.pi, 60)
    v = np.linspace(0.0, np.pi, 60)
    xs = radii[0] * np.outer(np.cos(u), np.sin(v))
    ys = radii[1] * np.outer(np.sin(u), np.sin(v))
    zs = radii[2] * np.outer(np.ones_like(u), np.cos(v))

    for i in range(len(xs)):
        for j in range(len(xs)):
            [xs[i,j],ys[i,j],zs[i,j]] = np.dot([xs[i,j],ys[i,j],zs[i,j]], rotation) + center

    return xs, ys, zs

# Plot: position estimate
fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(projection='3d')
ax.plot(estim_x, estim_y, estim_z, 'r.', markersize=3, label='estimate')
ax.plot(gt_x, gt_y, gt_z, 'k.', markersize=3, label='true')
ax.plot(gt_x[0], gt_y[0], gt_z[0], 'o', color='dimgrey', markersize=5, label='true initial position')
ax.plot(estim_x[0], estim_y[0], estim_z[0], 'o', color='orange', markersize=5, label='estimate initial position')
ax.set_xlabel('x, m')
ax.set_ylabel('y, m')
ax.set_zlabel('z, m')
ax.legend()
plt.title('Position Estimate')
# plt.savefig('./hw_myself/results/position_estimation.png')
plt.axis('equal')
plt.show()

# # Plot: position estimate with 3-sigma error
# fig = plt.figure(figsize=(6,6))
# ax = fig.add_subplot(projection='3d')
# for i in range(0, len(time), 4):
#     xs, ys, zs = compute3DEllipsoidCoordinates(estim_x[i], estim_y[i], estim_z[i], 3*cov_x[i], 3*cov_y[i], 3*cov_z[i])
#     ax.plot_surface(xs, ys, zs,  rstride=3, cstride=3, cmap='summer', alpha=0.8, shade=True)
# ax.plot(gt_x, gt_y, gt_z, 'k.', markersize=3, label='true')
# ax.legend()
# plt.title('Position Estimate with 3-Sigma Error')
# # plt.savefig('./hw_myself/results/error_covariance.png')
# plt.axis('equal')
# plt.show()

# # Plot: measurement
# fig = plt.figure()
# ax1 = fig.add_subplot(4, 1, 1)
# ax1.plot(time, meas_u_l)
# ax1.set_xlabel('time (s)')
# ax1.set_ylabel('u_l (pixel)')

# ax2 = fig.add_subplot(4, 1, 2)
# ax2.plot(time, meas_v_l)
# ax2.set_xlabel('time (s)')
# ax2.set_ylabel('v_l (pixel)')

# ax3 = fig.add_subplot(4, 1, 3)
# ax3.plot(time, meas_u_r)
# ax3.set_xlabel('time (s)')
# ax3.set_ylabel('u_r (pixel)')

# ax4 = fig.add_subplot(4, 1, 4)
# ax4.plot(time, meas_v_r)
# ax4.set_xlabel('time (s)')
# ax4.set_ylabel('v_r (pixel)')

# fig.suptitle('Sensor Measurement')
# # plt.savefig('./hw_myself/results/sensor_measurement.png')
# plt.show()

# Plot: estimation error
fig = plt.figure()
ax1 = fig.add_subplot(3, 1, 1)
ax1.plot(time, estim_x - gt_x)
# ax1.plot(time, 3*cov_x, 'k--')
# ax1.plot(time, -3*cov_x, 'k--')
ax1.set_xlabel('time (s)')
ax1.set_ylabel('x position error (m)')

ax2 = fig.add_subplot(3, 1, 2)
ax2.plot(time, estim_y - gt_y)
# ax2.plot(time, 3*cov_y, 'k--')
# ax2.plot(time, -3*cov_y, 'k--')
ax2.set_xlabel('time (s)')
ax2.set_ylabel('y position error (m)')

ax3 = fig.add_subplot(3, 1, 3)
ax3.plot(time, estim_z - gt_z)
# ax3.plot(time, 3*cov_z, 'k--')
# ax3.plot(time, -3*cov_z, 'k--')
ax3.set_xlabel('time (s)')
ax3.set_ylabel('z position error (m)')
fig.suptitle('Position Error')
# plt.savefig('./hw_myself/results/measurement_error.png')
plt.show()