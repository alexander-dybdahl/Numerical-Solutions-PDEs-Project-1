import numpy as np
import matplotlib.pyplot as plt

# Define the domain and grid parameters
x_min, x_max = 0, 1
y_min, y_max = 0, 1
N = 30 # number of grid points in each direction
dx = (x_max - x_min) / (N - 1)
dy = (y_max - y_min) / (N - 1)

# Define the boundary curves
gamma_1 = np.array([(x, 0) for x in np.linspace(x_min, x_max, N)])
gamma_2 = np.array([(0, y) for y in np.linspace(y_min, y_max, N)])
gamma_3 = np.array([(x, 1 - x**2) for x in np.linspace(x_min, x_max, N)])

# Define the fatten boundary distance
d = dx/2

# Create a meshgrid for plotting
X, Y = np.meshgrid(np.linspace(x_min, x_max, N), np.linspace(y_min, y_max, N))

# Define the inner grid points
inner_points = np.zeros((N, N), dtype=bool)
for i in range(N):
    for j in range(N):
        if Y[i,j] > gamma_1[j,1] and Y[i,j] > gamma_2[i,1] and Y[i,j] < 1 - X[i,j]**2:
            inner_points[i,j] = True

# Plot the domain and the boundary curves
fig, ax = plt.subplots(figsize=(6, 6))
ax.plot(gamma_1[:,0], gamma_1[:,1], 'r')
ax.plot(gamma_2[:,0], gamma_2[:,1], 'r')
ax.plot(gamma_3[:,0], gamma_3[:,1], 'r')

# Plot the inner grid points inside the domain in blue
ax.plot(X[inner_points], Y[inner_points], 'k.', markersize=3)

# Loop over all grid points to plot the normal lines to the boundary
for i in range(N):
    for j in range(N):
        if not inner_points[i, j]:
            continue
        xp, yp = j*dx, i*dy
        dist_to_gamma_3 = np.abs(gamma_3[:, 0] - xp) + np.abs(gamma_3[:, 1] - yp)
        idx = np.argmin(dist_to_gamma_3)
        x_new, y_new = gamma_3[idx, 0], gamma_3[idx, 1]
        ax.plot([xp, x_new], [yp, y_new], 'r--')

# Set axis limits and labels
ax.set_xlim([x_min, x_max])
ax.set_ylim([y_min, y_max])
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.show()