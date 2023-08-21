import matplotlib.pyplot as plt

# Load data from txt files
gt_u_l = [float(value) for line in open("/home/kblee/catkin_ws/src/flightmare/flightlib/include/flightlib/data/sensor_outputgt_u_l.txt") for value in line.split()]
gt_v_l = [float(value) for line in open("/home/kblee/catkin_ws/src/flightmare/flightlib/include/flightlib/data/sensor_outputgt_v_l.txt") for value in line.split()]
gt_u_r = [float(value) for line in open("/home/kblee/catkin_ws/src/flightmare/flightlib/include/flightlib/data/sensor_outputgt_u_r.txt") for value in line.split()]
gt_v_r = [float(value) for line in open("/home/kblee/catkin_ws/src/flightmare/flightlib/include/flightlib/data/sensor_outputgt_v_r.txt") for value in line.split()]

u_l = [float(value) for line in open("/home/kblee/catkin_ws/src/flightmare/flightlib/include/flightlib/data/sensor_outputu_l.txt") for value in line.split()]
v_l = [float(value) for line in open("/home/kblee/catkin_ws/src/flightmare/flightlib/include/flightlib/data/sensor_outputv_l.txt") for value in line.split()]
u_r = [float(value) for line in open("/home/kblee/catkin_ws/src/flightmare/flightlib/include/flightlib/data/sensor_outputu_r.txt") for value in line.split()]
v_r = [float(value) for line in open("/home/kblee/catkin_ws/src/flightmare/flightlib/include/flightlib/data/sensor_outputv_r.txt") for value in line.split()]

gt_depth = [float(line.strip()) for line in open("/home/kblee/catkin_ws/src/flightmare/flightlib/include/flightlib/data/sensor_outputgt_depth.txt")]
depth = [float(line.strip()) for line in open("/home/kblee/catkin_ws/src/flightmare/flightlib/include/flightlib/data/sensor_outputdepth.txt")]

time = [float(line.strip()) for line in open("/home/kblee/catkin_ws/src/flightmare/flightlib/include/flightlib/data/sensor_outputtime.txt")]

# Plotting
plt.figure(figsize=(10, 8))

# Plot System Output
plt.subplot(5, 1, 1)
plt.plot(time, u_l, label='estimated u_l')
plt.plot(time, gt_u_l, label='gt x')
plt.title('Step Response')
plt.xlabel('Time')
plt.legend()

plt.subplot(5, 1, 2)
plt.plot(time, v_l, label='estimated v_l')
plt.plot(time, gt_v_l,label='gt y')
plt.legend()

plt.subplot(5, 1, 3)
plt.plot(time, u_r, label='estimated u_r')
plt.plot(time, gt_u_r, label='gt z')
plt.legend()

plt.subplot(5, 1, 4)
plt.plot(time, v_r, label='estimated v_r')
plt.plot(time, gt_v_r, label='gt z')
plt.legend()

plt.subplot(5, 1, 5)
plt.plot(time, depth, label='estimated depth')
plt.plot(time, gt_depth, label='gt depth')
plt.legend()

plt.tight_layout()
plt.show()