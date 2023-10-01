import numpy as np
from numpy import linalg
import matplotlib.pyplot as plt
import argparse
import sys
from matplotlib.animation import FuncAnimation, PillowWriter


def load_data(data_dir, num_targets, num_trackers):
    # Load data from txt files
    time = np.array([float(line.strip()) for line in open(data_dir + "time.txt")])
    timesteps = time.shape[0]

    ego_pos = np.zeros([3, timesteps])
    ego_pos[0, :] = np.array([float(value) for line in open(data_dir + "ego_x.txt") for value in line.split()])
    ego_pos[1, :] = np.array([float(value) for line in open(data_dir + "ego_y.txt") for value in line.split()])
    ego_pos[2, :] = np.array([float(value) for line in open(data_dir + "ego_z.txt") for value in line.split()])

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

    return ego_pos, target_gt, target_estim, target_cov, tracker_gt, tracker_estim, tracker_cov, time


def plot_3d_ellipsoid(cx, cy, cz, x_axis, y_axis, z_axis, ax, target=True):
    """Plot the 3-d Ellipsoid ell on the Axes3D ax."""

    P = np.array([[x_axis,0,0],
                  [0,y_axis,0],
                  [0,0,z_axis]])

    _, radii, rotation = linalg.svd(P)

    # points on unit sphere
    u = np.linspace(0.0, 2.0 * np.pi, 100)
    v = np.linspace(0.0, np.pi, 100)
    z = radii[0] * np.outer(np.cos(u), np.sin(v))
    y = radii[1] * np.outer(np.sin(u), np.sin(v))
    x = radii[2] * np.outer(np.ones_like(u), np.cos(v))

    # transform points to ellipsoid
    for i in range(len(x)):
        for j in range(len(x)):
            x[i,j], y[i,j], z[i,j] = [cx, cy, cz] + np.dot(rotation, [x[i,j],y[i,j],z[i,j]])

    if target:
        ax.plot_wireframe(x, y, z,  rstride=10, cstride=10, color='#ff7575', alpha=0.2)
    else:
        ax.plot_wireframe(x, y, z,  rstride=10, cstride=10, color='#7a70ff', alpha=0.2)

# def update():

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default="/home/kblee/catkin_ws/src/flightmare/flightlib/include/flightlib/data/tracking_output/")
    parser.add_argument('--targets', type=int, default=1, help="The number of targets")
    parser.add_argument('--trackers', type=int, default=0, help="The number of trackers except itself")
    args = parser.parse_args()

    ego_pos, target_gt, target_estim, target_cov, tracker_gt, tracker_estim, tracker_cov, time = load_data(args.data_dir, args.targets, args.trackers)

    # Plot: position estimate
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(projection='3d')
    # Ego drone
    ax.plot(ego_pos[0,:], ego_pos[1,:], ego_pos[2,:], 'kx', markersize=3, label='true')
    for i in range(args.targets):
        ax.plot(target_gt[i,0,:], target_gt[i,1,:], target_gt[i,2,:], 'k.', markersize=3, label='true')
        ax.plot(target_estim[i,0,:], target_estim[i,1,:], target_estim[i,2,:], 'r.', markersize=3, label='estimate')
        ax.plot(target_gt[i,0,0], target_gt[i,1,0], target_gt[i,2,0], 'o', color='dimgrey', markersize=5, label='true initial position')
        ax.plot(target_estim[i,0,0], target_estim[i,1,0], target_estim[i,2,0], 'o', color='orange', markersize=5, label='estimate initial position')
    for i in range(args.trackers):
        ax.plot(tracker_gt[i,0,:], tracker_gt[i,1,:], tracker_gt[i,2,:], 'k.', markersize=3, label='true')
        ax.plot(tracker_estim[i,0,:], tracker_estim[i,1,:], tracker_estim[i,2,:], 'r.', markersize=3, label='estimate')
        ax.plot(tracker_gt[i,0,0], tracker_gt[i,1,0], tracker_gt[i,2,0], 'o', color='dimgrey', markersize=5, label='true initial position')
        ax.plot(tracker_estim[i,0,0], tracker_estim[i,1,0], tracker_estim[i,2,0], 'o', color='orange', markersize=5, label='estimate initial position')
    
    ax.set_xlabel('x, m')
    ax.set_ylabel('y, m')
    ax.set_zlabel('z, m')
    # ax.legend()
    plt.title('Position Estimate')
    # plt.savefig(args.data_dir + 'position_estimation.png')
    plt.axis('equal')
    plt.show()

    # # Plot: position estimate with 3-sigma error
    # fig = plt.figure(figsize=(6,6))
    # ax = fig.add_subplot(projection='3d')

    # for i in range(args.targets):
    #     for t in range(0, len(time), 10):
    #         plot_3d_ellipsoid(target_estim[i,0,t], target_estim[i,1,t], target_estim[i,2,t], 3*target_cov[i,0,t], 3*target_cov[i,1,t], 3*target_cov[i,2,t], ax)
    # for i in range(args.trackers):
    #     for t in range(0, len(time), 10):
    #         plot_3d_ellipsoid(tracker_estim[i,0,t], tracker_estim[i,1,t], tracker_estim[i,2,t], 3*tracker_cov[i,0,t], 3*tracker_cov[i,1,t], 3*tracker_cov[i,2,t], ax)

    # # ax.plot(gt_x, gt_y, gt_z, 'k.', markersize=3, label='true')
    # ax.legend()
    # plt.title('Position Estimate with 3-Sigma Error')
    # # plt.savefig('./hw_myself/results/error_covariance.png')
    # plt.axis('equal')
    # plt.show()

    # Show animation
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Update function for animation
    def update(t):
        ax.cla()
        
        # Ego drone
        ax.plot(ego_pos[0, t], ego_pos[1, t], ego_pos[2, t], 'kx', markersize=5, label='true')
        # Targets
        for i in range(args.targets):
            ax.plot(target_gt[i, 0, t], target_gt[i, 1, t], target_gt[i, 2, t], 'o', color='#fa0000', markersize=3, label='true')
            ax.plot(target_estim[i, 0, t], target_estim[i, 1, t], target_estim[i, 2, t], 'o', color='#ff7575', markersize=3, label='estimate')
            plot_3d_ellipsoid(target_estim[i, 0, t], target_estim[i, 1, t], target_estim[i, 2, t],
                              3*target_cov[i, 0, t], 3*target_cov[i, 1, t], 3*target_cov[i, 2, t], ax)
        # Trackers
        for i in range(args.trackers):
            ax.plot(tracker_gt[i, 0, t], tracker_gt[i, 1, t], tracker_gt[i, 2, t], 'o.', color='#1100fa', markersize=3, label='true')
            ax.plot(tracker_estim[i, 0, t], tracker_estim[i, 1, t], tracker_estim[i, 2, t], 'o', color='#7a70ff', markersize=3, label='estimate')

        ax.axis("equal")
        ax.set_title("Time[s]:" + str(round(time[t], 2)))
        ax.axes.set_xlim3d(left=-10, right=10)
        ax.axes.set_ylim3d(bottom=-10, top=10)
        ax.axes.set_zlim3d(bottom=0, top=20)

    # Set up the animation
    ani = FuncAnimation(fig, update, frames=len(time), interval=100, repeat=False)

    # Connect key event to figure
    fig.canvas.mpl_connect('key_press_event', lambda event: [exit(0) if event.key == 'escape' else None])

    # # Save as GIF
    # writer = PillowWriter(fps=20)  # Adjust fps (frames per second) as needed
    # ani.save(args.data_dir + 'animation.gif', writer=writer)

    plt.show()

if __name__ == "__main__":
    main()