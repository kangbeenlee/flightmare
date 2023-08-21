import matplotlib.pyplot as plt

# Load data from txt files
vx_des = [float(value) for line in open("/home/kblee/catkin_ws/src/flightmare/flightlib/include/flightlib/data/controller_output/vx_des.txt") for value in line.split()]
vy_des = [float(value) for line in open("/home/kblee/catkin_ws/src/flightmare/flightlib/include/flightlib/data/controller_output/vy_des.txt") for value in line.split()]
vz_des = [float(value) for line in open("/home/kblee/catkin_ws/src/flightmare/flightlib/include/flightlib/data/controller_output/vz_des.txt") for value in line.split()]
wz_des = [float(value) for line in open("/home/kblee/catkin_ws/src/flightmare/flightlib/include/flightlib/data/controller_output/wz_des.txt") for value in line.split()]
phi_des = [float(value) for line in open("/home/kblee/catkin_ws/src/flightmare/flightlib/include/flightlib/data/controller_output/phi_des.txt") for value in line.split()]
theta_des = [float(value) for line in open("/home/kblee/catkin_ws/src/flightmare/flightlib/include/flightlib/data/controller_output/theta_des.txt") for value in line.split()]

input_T = [float(value) for line in open("/home/kblee/catkin_ws/src/flightmare/flightlib/include/flightlib/data/controller_output/thrust.txt") for value in line.split()]
input_Mx = [float(value) for line in open("/home/kblee/catkin_ws/src/flightmare/flightlib/include/flightlib/data/controller_output/torque_x.txt") for value in line.split()]
input_My = [float(value) for line in open("/home/kblee/catkin_ws/src/flightmare/flightlib/include/flightlib/data/controller_output/torque_y.txt") for value in line.split()]
input_Mz = [float(value) for line in open("/home/kblee/catkin_ws/src/flightmare/flightlib/include/flightlib/data/controller_output/torque_z.txt") for value in line.split()]

output_vx = [float(value) for line in open("/home/kblee/catkin_ws/src/flightmare/flightlib/include/flightlib/data/controller_output/vx_o.txt") for value in line.split()]
output_vy = [float(value) for line in open("/home/kblee/catkin_ws/src/flightmare/flightlib/include/flightlib/data/controller_output/vy_o.txt") for value in line.split()]
output_vz = [float(value) for line in open("/home/kblee/catkin_ws/src/flightmare/flightlib/include/flightlib/data/controller_output/vz_o.txt") for value in line.split()]
output_wz = [float(value) for line in open("/home/kblee/catkin_ws/src/flightmare/flightlib/include/flightlib/data/controller_output/wz_o.txt") for value in line.split()]
output_phi = [float(value) for line in open("/home/kblee/catkin_ws/src/flightmare/flightlib/include/flightlib/data/controller_output/phi_o.txt") for value in line.split()]
output_theta = [float(value) for line in open("/home/kblee/catkin_ws/src/flightmare/flightlib/include/flightlib/data/controller_output/theta_o.txt") for value in line.split()]
output_psi = [float(value) for line in open("/home/kblee/catkin_ws/src/flightmare/flightlib/include/flightlib/data/controller_output/psi_o.txt") for value in line.split()]

time = [float(line.strip()) for line in open("/home/kblee/catkin_ws/src/flightmare/flightlib/include/flightlib/data/controller_output/time.txt")]


# Plotting
plt.figure(figsize=(12, 11))

# Plot System Output
plt.subplot(6, 2, 1)
plt.plot(time, output_vx, label='system_vx')
plt.plot(time, vx_des, linestyle='--', label='desired_vx')
plt.title('Step Response')
plt.xlabel('Time')
plt.legend()

# Plot System Output
plt.subplot(6, 2, 3)
plt.plot(time, output_vy, label='system_vy')
plt.plot(time, vy_des, linestyle='--', label='desired_vy')
plt.legend()

# Plot System Output
plt.subplot(6, 2, 5)
plt.plot(time, output_vz, label='system_vz')
plt.plot(time, vz_des, linestyle='--', label='desired_vz')
plt.legend()

# Plot System Output
plt.subplot(6, 2, 7)
plt.plot(time, output_wz, label='system_wz')
plt.plot(time, wz_des, linestyle='--', label='desired_wz')
plt.legend()

# Plot System Output
plt.subplot(6, 2, 9)
plt.plot(time, output_phi, label='system_phi')
plt.plot(time, phi_des, linestyle='--', label='desired_phi')
plt.legend()

# Plot System Output
plt.subplot(6, 2, 11)
plt.plot(time, output_theta, label='system_theta')
plt.plot(time, theta_des, linestyle='--', label='desired_theta')
plt.legend()

# Plot System Output
plt.subplot(6, 2, 12)
plt.plot(time, output_psi, label='system_yaw')
plt.legend()

# Plot Control Input
plt.subplot(6, 2, 2)
plt.plot(time, input_T, label='Thrust', color='green')
plt.title('Control Input')
plt.xlabel('Time')
plt.legend()

# Plot Control Input
plt.subplot(6, 2, 4)
plt.plot(time, input_Mx, label='Torque_x', color='green')
plt.legend()

# Plot Control Input
plt.subplot(6, 2, 6)
plt.plot(time, input_My, label='Torque_y', color='green')
plt.legend()

# Plot Control Input
plt.subplot(6, 2, 8)
plt.plot(time, input_Mz, label='Torque_z', color='green')
plt.legend()

plt.tight_layout()
plt.show()