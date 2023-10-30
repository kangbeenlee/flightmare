import numpy as np
from numpy import linalg
import matplotlib.pyplot as plt
import argparse
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from utils import R_x, R_y, R_z, quaternion_to_rotation_matrix, rotation_matrix
import os
import sys



def load_data(data_dir, num_targets, num_trackers):
    # Load data from txt files
    time = np.array([float(line.strip()) for line in open(data_dir + "time.txt")])
    timesteps = time.shape[0]

    ego_pos = np.zeros([3, timesteps])
    ego_pos[0, :] = np.array([float(value) for line in open(data_dir + "ego_x.txt") for value in line.split()])
    ego_pos[1, :] = np.array([float(value) for line in open(data_dir + "ego_y.txt") for value in line.split()])
    ego_pos[2, :] = np.array([float(value) for line in open(data_dir + "ego_z.txt") for value in line.split()])
    ego_orien = np.zeros([4, timesteps])
    ego_orien[0, :] = np.array([float(value) for line in open(data_dir + "ego_qw.txt") for value in line.split()])
    ego_orien[1, :] = np.array([float(value) for line in open(data_dir + "ego_qx.txt") for value in line.split()])
    ego_orien[2, :] = np.array([float(value) for line in open(data_dir + "ego_qy.txt") for value in line.split()])
    ego_orien[3, :] = np.array([float(value) for line in open(data_dir + "ego_qz.txt") for value in line.split()])

    target_gt = np.zeros([num_targets, 3, timesteps])
    target_estim = np.zeros([num_targets, 3, timesteps])
    target_cov = np.zeros([num_targets, 3, 3, timesteps])

    tracker_gt = np.zeros([num_trackers, 3, timesteps])
    tracker_estim = np.zeros([num_trackers, 3, timesteps])
    tracker_cov = np.zeros([num_trackers, 3, timesteps])

    for i in range(num_targets):
        target_gt[i, 0, :] = np.array([float(value) for line in open(data_dir + "target_gt_x_" + str(i) + ".txt") for value in line.split()])
        target_gt[i, 1, :] = np.array([float(value) for line in open(data_dir + "target_gt_y_" + str(i) + ".txt") for value in line.split()])
        target_gt[i, 2, :] = np.array([float(value) for line in open(data_dir + "target_gt_z_" + str(i) + ".txt") for value in line.split()])

        target_estim[i, 0, :] = np.array([float(value) for line in open(data_dir + "target_estim_x_" + str(i) + ".txt") for value in line.split()])
        target_estim[i, 1, :] = np.array([float(value) for line in open(data_dir + "target_estim_y_" + str(i) + ".txt") for value in line.split()])
        target_estim[i, 2, :] = np.array([float(value) for line in open(data_dir + "target_estim_z_" + str(i) + ".txt") for value in line.split()])

        target_cov[i, 0, 0, :] = np.array([float(value) for line in open(data_dir + "target_cov_xx_" + str(i) + ".txt") for value in line.split()])
        target_cov[i, 0, 1, :] = np.array([float(value) for line in open(data_dir + "target_cov_xy_" + str(i) + ".txt") for value in line.split()])
        target_cov[i, 0, 2, :] = np.array([float(value) for line in open(data_dir + "target_cov_xz_" + str(i) + ".txt") for value in line.split()])
        target_cov[i, 1, 0, :] = np.array([float(value) for line in open(data_dir + "target_cov_yx_" + str(i) + ".txt") for value in line.split()])
        target_cov[i, 1, 1, :] = np.array([float(value) for line in open(data_dir + "target_cov_yy_" + str(i) + ".txt") for value in line.split()])
        target_cov[i, 1, 2, :] = np.array([float(value) for line in open(data_dir + "target_cov_yz_" + str(i) + ".txt") for value in line.split()])
        target_cov[i, 2, 0, :] = np.array([float(value) for line in open(data_dir + "target_cov_zx_" + str(i) + ".txt") for value in line.split()])
        target_cov[i, 2, 1, :] = np.array([float(value) for line in open(data_dir + "target_cov_zy_" + str(i) + ".txt") for value in line.split()])
        target_cov[i, 2, 2, :] = np.array([float(value) for line in open(data_dir + "target_cov_zz_" + str(i) + ".txt") for value in line.split()])

    for i in range(num_trackers):
        tracker_gt[i, 0, :] = np.array([float(value) for line in open(data_dir + "tracker_gt_x_" + str(i) + ".txt") for value in line.split()])
        tracker_gt[i, 1, :] = np.array([float(value) for line in open(data_dir + "tracker_gt_y_" + str(i) + ".txt") for value in line.split()])
        tracker_gt[i, 2, :] = np.array([float(value) for line in open(data_dir + "tracker_gt_z_" + str(i) + ".txt") for value in line.split()])

        tracker_estim[i, 0, :] = np.array([float(value) for line in open(data_dir + "tracker_estim_x_" + str(i) + ".txt") for value in line.split()])
        tracker_estim[i, 1, :] = np.array([float(value) for line in open(data_dir + "tracker_estim_y_" + str(i) + ".txt") for value in line.split()])
        tracker_estim[i, 2, :] = np.array([float(value) for line in open(data_dir + "tracker_estim_z_" + str(i) + ".txt") for value in line.split()])

        tracker_cov[i, 0, :] = np.array([float(value) for line in open(data_dir + "tracker_cov_x_" + str(i) + ".txt") for value in line.split()])
        tracker_cov[i, 1, :] = np.array([float(value) for line in open(data_dir + "tracker_cov_y_" + str(i) + ".txt") for value in line.split()])
        tracker_cov[i, 2, :] = np.array([float(value) for line in open(data_dir + "tracker_cov_z_" + str(i) + ".txt") for value in line.split()])

    return ego_pos, ego_orien, target_gt, target_estim, target_cov, tracker_gt, tracker_estim, tracker_cov, time


def plot_3d_ellipsoid(mean, cov, ax, target=True):
    """Plot the 3-d Ellipsoid ell on the Axes3D ax."""

    # Compute the eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # Using the eigenvalues to generate radii for the ellipsoid
    radii = np.sqrt(eigenvalues)

    # Generate data for the ellipsoid surface
    u = np.linspace(0.0, 2.0 * np.pi, 100)
    v = np.linspace(0.0, np.pi, 100)
    x = radii[0] * np.outer(np.cos(u), np.sin(v))
    y = radii[1] * np.outer(np.sin(u), np.sin(v))
    z = radii[2] * np.outer(np.ones_like(u), np.cos(v))
    for i in range(len(u)):
        for j in range(len(v)):
            [x[i, j], y[i, j], z[i, j]] = np.dot(eigenvectors, [x[i, j], y[i, j], z[i, j]])

    # add center coordinates
    x += mean[0]
    y += mean[1]
    z += mean[2]

    if target:
        ax.plot_wireframe(x, y, z,  rstride=10, cstride=10, color='#ff7575', alpha=0.2)
    else:
        ax.plot_wireframe(x, y, z,  rstride=10, cstride=10, color='#7a70ff', alpha=0.2)


def plot_drone(ax, center, R_b):
    x, y, z = center[0], center[1], center[2]
    rotor_distance = 0.25
    
    # Central body
    ax.scatter(x, y, z, c='k', s=10)

    rotor_1 = R_b @ np.array([ rotor_distance,  rotor_distance, 0]) + np.array([x, y, z])
    rotor_2 = R_b @ np.array([-rotor_distance,  rotor_distance, 0]) + np.array([x, y, z])
    rotor_3 = R_b @ np.array([ rotor_distance, -rotor_distance, 0]) + np.array([x, y, z])
    rotor_4 = R_b @ np.array([-rotor_distance, -rotor_distance, 0]) + np.array([x, y, z])

    ax.scatter([rotor_1[0], rotor_2[0], rotor_3[0], rotor_4[0]],
               [rotor_1[1], rotor_2[1], rotor_3[1], rotor_4[1]],
               [rotor_1[2], rotor_2[2], rotor_3[2], rotor_4[2]], c='g', s=10)
    
    # Arms
    arm_1 = R_b @ np.array([ rotor_distance,  rotor_distance, 0]) + np.array([x, y, z])
    arm_2 = R_b @ np.array([-rotor_distance,  rotor_distance, 0]) + np.array([x, y, z])
    arm_3 = R_b @ np.array([ rotor_distance, -rotor_distance, 0]) + np.array([x, y, z])
    arm_4 = R_b @ np.array([-rotor_distance, -rotor_distance, 0]) + np.array([x, y, z])

    ax.plot([x, arm_1[0]], [y, arm_1[1]], [z, arm_1[2]], c='k')
    ax.plot([x, arm_2[0]], [y, arm_2[1]], [z, arm_2[2]], c='k')
    ax.plot([x, arm_3[0]], [y, arm_3[1]], [z, arm_3[2]], c='k')
    ax.plot([x, arm_4[0]], [y, arm_4[1]], [z, arm_4[2]], c='k')
    
    # Drone's body frame
    colors = ['r', 'g', 'b']  # RGB for XYZ
    for i, color in enumerate(colors):
        ax.quiver(x, y, z, R_b[0, i], R_b[1, i], R_b[2, i], length=0.4, color=color)


def plot_ego(center, R_b, ax):
    ax.plot(center[0], center[1], center[2], 'kx', markersize=3, label='true')

    # Plot drone
    plot_drone(ax, center, R_b)

    # Image window cooridinates w.r.t. camera frame
    scale = 0.5
    fov_scale = 4.0

    # 2.88 1.62 2.2 (focal length) / (mm scale)
    corners = np.array([[ 2.88,  1.62,  2.2],
                        [ 2.88, -1.62,  2.2],
                        [-2.88,  1.62,  2.2],
                        [-2.88, -1.62,  2.2]]) * scale
    front = np.zeros([4, 3])

    # Plot front camera field of view
    front_fov = np.zeros([4, 3])
    corners_fov = corners * fov_scale

    # Transformation from left camera to body
    R1 = R_x(-np.pi/2) @ R_y(np.pi/2)
    t1 = np.array([0.1, 0.06, 0.0])
    
    # From camera frame to body frame
    for i in range(4):
        front[i] = R1 @ corners[i] + t1
        front_fov[i] = R1 @ corners_fov[i] + t1

    # Plot camera image
    for i in range(4):
        front[i] = R_b @ front[i] + center # front camera image w.r.t. world
        front_fov[i] = R_b @ front_fov[i] + center

    t1 = R_b @ t1 + center # front camera origin w.r.t. world

    # Plot front camera field of view
    v = np.vstack((front_fov, t1))
    vertices = [v[[0, 2, 4]], v[[2, 3, 4]], v[[3, 1, 4]], v[[1, 0, 4]]]
    for verts in vertices:
        ax.add_collection3d(Poly3DCollection([verts], alpha=.15, linewidths=1, edgecolors='#e8ebff'))

    for origin, image in zip([t1], [front]):
        t_l, t_r, b_l, b_r = image
        ax.plot(*zip(t_l, t_r), color='black', linewidth=0.5)
        ax.plot(*zip(t_l, b_l), color='black', linewidth=0.5)
        ax.plot(*zip(t_r, b_r), color='black', linewidth=0.5)
        ax.plot(*zip(b_r, b_l), color='black', linewidth=0.5)
        ax.plot(*zip(origin, t_l), color='grey', linewidth=0.5)
        ax.plot(*zip(origin, t_r), color='grey', linewidth=0.5)
        ax.plot(*zip(origin, b_l), color='grey', linewidth=0.5)
        ax.plot(*zip(origin, b_r), color='grey', linewidth=0.5)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default="/home/kblee/catkin_ws/src/flightmare/flightlib/include/flightlib/data/tracking_output/")
    parser.add_argument('--targets', type=int, default=1, help="The number of targets")
    parser.add_argument('--trackers', type=int, default=0, help="The number of other trackers except itself (total # of tracker - 1)")
    parser.add_argument('--tracker_id', type=int, default=0, help="The id of ego tracker (agent)")
    args = parser.parse_args()

    data_path = os.path.join(args.data_dir, 'tracker_'+ str(args.tracker_id) + '/')
    ego_pos, ego_orien, target_gt, target_estim, target_cov, tracker_gt, tracker_estim, tracker_cov, time = load_data(data_path, args.targets, args.trackers)

    # # Compute average target covariance norm graph
    # avg_cov = np.zeros([time.shape[0]])
    # for i in range(args.targets):
    #     avg_cov += np.sqrt(target_cov[i, 0, :] ** 2 + target_cov[i, 1, :] ** 2 + target_cov[i, 2, :] ** 2)
    # avg_cov /= args.targets
    # avg_cov = np.log(avg_cov)

    # plt.figure(figsize=(10, 5))
    # plt.plot(avg_cov)
    # plt.title('Average Covariance from tracker {}'.format(args.tracker_id))
    # plt.xlabel('time')
    # plt.ylabel('average covariance')
    # plt.show()

    # Show animation
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Update function for animation
    def update(t):
        ax.cla()
        
        # Rotation from body to world
        R_b = quaternion_to_rotation_matrix(ego_orien[0, t], ego_orien[1, t], ego_orien[2, t], ego_orien[3, t])

        # Ego drone
        plot_ego(ego_pos[:, t], R_b, ax)

        # total_cov_norm = []

        # Targets and trackers
        for i in range(args.targets):
            ax.plot(target_gt[i, 0, t], target_gt[i, 1, t], target_gt[i, 2, t], 'o', color='#fa0000', markersize=3, label='true')
            ax.plot(target_estim[i, 0, t], target_estim[i, 1, t], target_estim[i, 2, t], 'o', color='#ff7575', markersize=3, label='estimate')
            plot_3d_ellipsoid(target_estim[i, :, t], target_cov[i, :, :, t], ax)

            # total_cov_norm.append(np.sqrt(target_cov[i, 0, t]**2 + target_cov[i, 1, t]**2 + target_cov[i, 2, t]**2))

        # avg_cov_norm = np.sum(total_cov_norm)/args.targets
        # cov_reward = np.exp(-0.001 * (avg_cov_norm ** 5))
        # cov_reward_v2 = np.exp(-0.01 * (avg_cov_norm ** 3))
        # print("all cov      :", np.around(np.array(total_cov_norm), 3))
        # print("avg cov      :", avg_cov_norm)
        # print("avg cov c2   :", cov_reward_v2)
        # print("cov_reward   :", cov_reward)

        for i in range(args.trackers):
            ax.plot(tracker_gt[i, 0, t], tracker_gt[i, 1, t], tracker_gt[i, 2, t], 'o', color='#1100fa', markersize=3, label='true')
            ax.plot(tracker_estim[i, 0, t], tracker_estim[i, 1, t], tracker_estim[i, 2, t], 'o', color='#7a70ff', markersize=3, label='estimate')
            plot_3d_ellipsoid(tracker_estim[i, 0, t], tracker_estim[i, 1, t], tracker_estim[i, 2, t],
                              3*np.sqrt(tracker_cov[i, 0, t]), 3*np.sqrt(tracker_cov[i, 1, t]), 3*np.sqrt(tracker_cov[i, 2, t]), ax, target=False)

        ax.axes.set_xlim3d(left=-10, right=10)
        ax.axes.set_ylim3d(bottom=-10, top=10)
        ax.axes.set_zlim3d(bottom=0, top=10)
        ax.set_aspect('equal')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        # ax.view_init(90, -90)  # 90 degrees elevation for top-down view, -90 degrees azimuth for proper orientation
        ax.set_title("Time[s]: {:.2f}".format(time[t]))


    # Set up the animation
    ani = FuncAnimation(fig, update, frames=500, interval=10, repeat=False)

    # Connect key event to figure
    fig.canvas.mpl_connect('key_press_event', lambda event: [exit(0) if event.key == 'escape' else None])

    # # Save as GIF
    # writer = PillowWriter(fps=20)  # Adjust fps (frames per second) as needed
    # ani.save(args.data_dir + 'ego_{}.gif'.format(args.tracker_id), writer=writer)

    plt.show()

if __name__ == "__main__":
    main()