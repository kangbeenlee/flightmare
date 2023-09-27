import matplotlib.pyplot as plt

# Load data from txt files
gt_u_l = [float(value) for line in open("/home/kblee/catkin_ws/src/flightmare/flightlib/include/flightlib/data/sensor_output/gt_u_l.txt") for value in line.split()]
gt_v_l = [float(value) for line in open("/home/kblee/catkin_ws/src/flightmare/flightlib/include/flightlib/data/sensor_output/gt_v_l.txt") for value in line.split()]
gt_u_r = [float(value) for line in open("/home/kblee/catkin_ws/src/flightmare/flightlib/include/flightlib/data/sensor_output/gt_u_r.txt") for value in line.split()]
gt_v_r = [float(value) for line in open("/home/kblee/catkin_ws/src/flightmare/flightlib/include/flightlib/data/sensor_output/gt_v_r.txt") for value in line.split()]

u_l = [float(value) for line in open("/home/kblee/catkin_ws/src/flightmare/flightlib/include/flightlib/data/sensor_output/u_l.txt") for value in line.split()]
v_l = [float(value) for line in open("/home/kblee/catkin_ws/src/flightmare/flightlib/include/flightlib/data/sensor_output/v_l.txt") for value in line.split()]
u_r = [float(value) for line in open("/home/kblee/catkin_ws/src/flightmare/flightlib/include/flightlib/data/sensor_output/u_r.txt") for value in line.split()]
v_r = [float(value) for line in open("/home/kblee/catkin_ws/src/flightmare/flightlib/include/flightlib/data/sensor_output/v_r.txt") for value in line.split()]

gt_x = [float(value) for line in open("/home/kblee/catkin_ws/src/flightmare/flightlib/include/flightlib/data/sensor_output/gt_x.txt") for value in line.split()]
gt_y = [float(value) for line in open("/home/kblee/catkin_ws/src/flightmare/flightlib/include/flightlib/data/sensor_output/gt_y.txt") for value in line.split()]
gt_z = [float(value) for line in open("/home/kblee/catkin_ws/src/flightmare/flightlib/include/flightlib/data/sensor_output/gt_z.txt") for value in line.split()]

x = [float(value) for line in open("/home/kblee/catkin_ws/src/flightmare/flightlib/include/flightlib/data/sensor_output/x.txt") for value in line.split()]
y = [float(value) for line in open("/home/kblee/catkin_ws/src/flightmare/flightlib/include/flightlib/data/sensor_output/y.txt") for value in line.split()]
z = [float(value) for line in open("/home/kblee/catkin_ws/src/flightmare/flightlib/include/flightlib/data/sensor_output/z.txt") for value in line.split()]

time = [float(line.strip()) for line in open("/home/kblee/catkin_ws/src/flightmare/flightlib/include/flightlib/data/sensor_output/time.txt")]

# Plotting
plt.figure(figsize=(14, 8))

# Plot System Output
plt.subplot(4, 2, 1)
plt.plot(time, u_l, label='sensor u_l')
plt.plot(time, gt_u_l, label='gt u_l')
plt.title('Pixel Ouput')
plt.xlabel('Time')
plt.legend()

plt.subplot(4, 2, 3)
plt.plot(time, v_l, label='sensor v_l')
plt.plot(time, gt_v_l,label='gt v_l')
plt.legend()

plt.subplot(4, 2, 5)
plt.plot(time, u_r, label='sensor u_r')
plt.plot(time, gt_u_r,label='gt u_r')
plt.legend()

plt.subplot(4, 2, 7)
plt.plot(time, v_r, label='sensor v_r')
plt.plot(time, gt_v_r,label='gt v_r')
plt.legend()

# Plot System Output
plt.subplot(4, 2, 2)
plt.plot(time, x, label='sensor x')
plt.plot(time, gt_x, label='gt x')
plt.title('Position Ouput')
plt.xlabel('Time')
plt.legend()

plt.subplot(4, 2, 4)
plt.plot(time, y, label='sensor y')
plt.plot(time, gt_y,label='gt y')
plt.legend()

plt.subplot(4, 2, 6)
plt.plot(time, z, label='sensor z')
plt.plot(time, gt_z,label='gt z')
plt.legend()

plt.tight_layout()
plt.show()