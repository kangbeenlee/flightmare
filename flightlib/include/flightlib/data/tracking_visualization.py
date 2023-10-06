import numpy as np
from numpy import linalg
import matplotlib.pyplot as plt
import argparse
import sys
from matplotlib.animation import FuncAnimation, PillowWriter
from utils import R_x, R_y, R_z, quaternion_to_rotation_matrix



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
    target_cov = np.zeros([num_targets, 3, timesteps])

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

        target_cov[i, 0, :] = np.array([float(value) for line in open(data_dir + "target_cov_x_" + str(i) + ".txt") for value in line.split()])
        target_cov[i, 1, :] = np.array([float(value) for line in open(data_dir + "target_cov_y_" + str(i) + ".txt") for value in line.split()])
        target_cov[i, 2, :] = np.array([float(value) for line in open(data_dir + "target_cov_z_" + str(i) + ".txt") for value in line.split()])

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


def plot_3d_ellipsoid(cx, cy, cz, x_axis, y_axis, z_axis, ax, target=True):
    """Plot the 3-d Ellipsoid ell on the Axes3D ax."""
    
    # points on unit sphere
    u = np.linspace(0.0, 2.0 * np.pi, 100)
    v = np.linspace(0.0, np.pi, 100)
    x = x_axis * np.outer(np.cos(u), np.sin(v))
    y = y_axis * np.outer(np.sin(u), np.sin(v))
    z = z_axis * np.outer(np.ones_like(u), np.cos(v))

    # add center coordinates
    x += cx
    y += cy
    z += cz

    if target:
        ax.plot_wireframe(x, y, z,  rstride=10, cstride=10, color='#ff7575', alpha=0.2)
    else:
        ax.plot_wireframe(x, y, z,  rstride=10, cstride=10, color='#7a70ff', alpha=0.2)


def plot_drone(ax, x, y, z, R_b):
    rotor_distance = 0.25
    
    # Central body
    ax.scatter(x, y, z, c='k', s=10)
    
    # Rotors
    # rotor_1 = R_b @ np.array([x + rotor_distance, y + rotor_distance, z])
    # rotor_2 = R_b @ np.array([x - rotor_distance, y + rotor_distance, z])
    # rotor_3 = R_b @ np.array([x + rotor_distance, y - rotor_distance, z])
    # rotor_4 = R_b @ np.array([x - rotor_distance, y - rotor_distance, z])

    rotor_1 = R_b @ np.array([ rotor_distance,  rotor_distance, 0]) + np.array([x, y, z])
    rotor_2 = R_b @ np.array([-rotor_distance,  rotor_distance, 0]) + np.array([x, y, z])
    rotor_3 = R_b @ np.array([ rotor_distance, -rotor_distance, 0]) + np.array([x, y, z])
    rotor_4 = R_b @ np.array([-rotor_distance, -rotor_distance, 0]) + np.array([x, y, z])

    ax.scatter([rotor_1[0], rotor_2[0], rotor_3[0], rotor_4[0]],
               [rotor_1[1], rotor_2[1], rotor_3[1], rotor_4[1]],
               [rotor_1[2], rotor_2[2], rotor_3[2], rotor_4[2]], c='g', s=10)
    
    # Arms
    arm_1 = R_b @ np.array([ rotor_distance,  rotor_distance, z]) + np.array([x, y, z])
    arm_2 = R_b @ np.array([-rotor_distance,  rotor_distance, z]) + np.array([x, y, z])
    arm_3 = R_b @ np.array([ rotor_distance, -rotor_distance, z]) + np.array([x, y, z])
    arm_4 = R_b @ np.array([-rotor_distance, -rotor_distance, z]) + np.array([x, y, z])

    ax.plot([x, arm_1[0]], [y, arm_1[1]], [z, z], c='k')
    ax.plot([x, arm_2[0]], [y, arm_2[1]], [z, z], c='k')
    ax.plot([x, arm_3[0]], [y, arm_3[1]], [z, z], c='k')
    ax.plot([x, arm_4[0]], [y, arm_4[1]], [z, z], c='k')
    
    # Drone's body frame
    colors = ['r', 'g', 'b']  # RGB for XYZ
    for i, color in enumerate(colors):
        ax.quiver(x, y, z, R_b[0, i], R_b[1, i], R_b[2, i], length=0.4, color=color)


def plot_ego(x, y, z, qw, qx, qy, qz, ax):
    ax.plot(x, y, z, 'kx', markersize=3, label='true')
    center = np.array([x, y, z])

    # Plot drone
    R_b = quaternion_to_rotation_matrix(qw, qx, qy, qz) # body orientation
    plot_drone(ax, x, y, z, R_b)

    # Image window cooridinates w.r.t. camera frame
    scale = 2.0
    corners = np.array([[0.32, 0.32,  0.32],
                        [0.32, 0.32, -0.32],
                        [0.32, -0.32,  0.32],
                        [0.32, -0.32, -0.32]]) * scale
    front, left, right = np.zeros([4, 3]), np.zeros([4, 3]), np.zeros([4, 3])

    # From body frame origin to (left) camera frame origin
    t1 = np.array([0.1, 0.06, 0.0])
    t2 = np.array(R_z(2/3*np.pi) @ t1).squeeze()
    t3 = np.array(R_z(-2/3*np.pi) @ t1).squeeze()

    R1 = np.eye(3)
    R2 = R_z(2/3*np.pi)
    R3 = R_z(-2/3*np.pi)
    
    # from camera frame to body frame
    for i in range(4):
        front[i] = R1 @ corners[i] + t1
        left[i] = R2 @ corners[i] + t2
        right[i] = R3 @ corners[i] + t3
    
    # Plot camera image
    for i in range(4):
        front[i] = R_b @ front[i] + center # front camera image w.r.t. world
        left[i] = R_b @ left[i] + center
        right[i] = R_b @ right[i] + center
    t1 = R_b @ t1 + center # front camera origin w.r.t. world
    t2 = R_b @ t2 + center
    t3 = R_b @ t3 + center

    for origin, image in zip([t1, t2, t3], [front, left, right]):
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
    parser.add_argument('--targets', type=int, default=3, help="The number of targets")
    parser.add_argument('--trackers', type=int, default=2, help="The number of trackers except itself")
    args = parser.parse_args()

    ego_pos, ego_orien, target_gt, target_estim, target_cov, tracker_gt, tracker_estim, tracker_cov, time = load_data(args.data_dir, args.targets, args.trackers)

    # Plot: position estimate
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(projection='3d')

    # Ego drone
    ax.plot(ego_pos[0,:], ego_pos[1,:], ego_pos[2,:], 'kx', linewidth=3)

    # Targets and trackers
    for i in range(args.targets):
        ax.plot(target_gt[i,0,:], target_gt[i,1,:], target_gt[i,2,:], '.', color='#6e0000', markersize=3, label='true', alpha=0.1)
        ax.plot(target_estim[i,0,:], target_estim[i,1,:], target_estim[i,2,:], '.', color='#ff7575', markersize=3, label='estimate')
        ax.plot(target_gt[i,0,0], target_gt[i,1,0], target_gt[i,2,0], '.', color='red', markersize=5, label='true initial position')
    for i in range(args.trackers):
        ax.plot(tracker_gt[i,0,:], tracker_gt[i,1,:], tracker_gt[i,2,:], '.', color='#07006e', markersize=3, label='true', alpha=0.1)
        ax.plot(tracker_estim[i,0,:], tracker_estim[i,1,:], tracker_estim[i,2,:], '.', color='#7a70ff', markersize=3, label='estimate')
        ax.plot(tracker_gt[i,0,0], tracker_gt[i,1,0], tracker_gt[i,2,0], '.', color='blue', markersize=5, label='true initial position')
    
    ax.set_xlabel('x, m')
    ax.set_ylabel('y, m')
    ax.set_zlabel('z, m')
    plt.title('Position Estimate')
    # plt.savefig(args.data_dir + 'position_estimation.png')
    plt.axis('equal')
    plt.show()

    # Show animation
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Update function for animation
    def update(t):
        ax.cla()
        
        # Ego drone
        plot_ego(ego_pos[0, t], ego_pos[1, t], ego_pos[2, t], ego_orien[0, t], ego_orien[1, t], ego_orien[2, t], ego_orien[3, t], ax)

        # Targets and trackers
        for i in range(args.targets):
            ax.plot(target_gt[i, 0, t], target_gt[i, 1, t], target_gt[i, 2, t], 'o', color='#fa0000', markersize=3, label='true')
            ax.plot(target_estim[i, 0, t], target_estim[i, 1, t], target_estim[i, 2, t], 'o', color='#ff7575', markersize=3, label='estimate')
            plot_3d_ellipsoid(target_estim[i, 0, t], target_estim[i, 1, t], target_estim[i, 2, t],
                              3*target_cov[i, 0, t], 3*target_cov[i, 1, t], 3*target_cov[i, 2, t], ax)
        for i in range(args.trackers):
            ax.plot(tracker_gt[i, 0, t], tracker_gt[i, 1, t], tracker_gt[i, 2, t], 'o', color='#1100fa', markersize=3, label='true')
            ax.plot(tracker_estim[i, 0, t], tracker_estim[i, 1, t], tracker_estim[i, 2, t], 'o', color='#7a70ff', markersize=3, label='estimate')
            plot_3d_ellipsoid(tracker_estim[i, 0, t], tracker_estim[i, 1, t], tracker_estim[i, 2, t],
                              3*tracker_cov[i, 0, t], 3*tracker_cov[i, 1, t], 3*tracker_cov[i, 2, t], ax, target=False)

        ax.axes.set_xlim3d(left=-10, right=10)
        ax.axes.set_ylim3d(bottom=-10, top=10)
        ax.axes.set_zlim3d(bottom=0, top=10)
        ax.set_aspect('equal')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_title("Time[s]:" + str(round(time[t], 2)))

    # Set up the animation
    ani = FuncAnimation(fig, update, frames=len(time), interval=1, repeat=False)

    # Connect key event to figure
    fig.canvas.mpl_connect('key_press_event', lambda event: [exit(0) if event.key == 'escape' else None])

    # # Save as GIF
    # writer = PillowWriter(fps=20)  # Adjust fps (frames per second) as needed
    # ani.save(args.data_dir + 'animation.gif', writer=writer)

    plt.show()

if __name__ == "__main__":
    main()