import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load the data from the file as a NumPy array
data = np.loadtxt("scripts/logs/log_20240603-201311.dat")

# Extract xc and xd from the data
# Assuming the columns are in the following order:
# time, gt, dt, xc (3 values), xd (3 values), xd_dot (3 values), qdot (3 values), q (3 values)
xc = data[:, 2:5]  # Extract columns for xc
xd = data[:, 5:8]  # Extract columns for xd


# Re-examine the assumptions and re-plot both xc and xd on the same plot to compare trajectories

# Create a combined 3D plot for xc and xd
fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(10, 8))

# Plot xc
ax.plot(xc[:, 0], xc[:, 1], xc[:, 2], label='xc', marker='o')
# Plot xd
ax.plot(xd[:, 0], xd[:, 1], xd[:, 2], label='xd', marker='^')

ax.set_title('Combined 3D Trajectories of xc and xd')
ax.set_xlabel('X Axis')
ax.set_ylabel('Y Axis')
ax.set_zlabel('Z Axis')
ax.legend()

plt.show()


# Create 3D plots for xc and xd
fig = plt.figure(figsize=(14, 7))

# Plot for xc
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot(xc[:, 0], xc[:, 1], xc[:, 2])
ax1.set_title('3D Trajectory of xc')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')

# Plot for xd
ax2 = fig.add_subplot(122, projection='3d')
ax2.plot(xd[:, 0], xd[:, 1], xd[:, 2])
ax2.set_title('3D Trajectory of xd')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Z')

plt.show()
