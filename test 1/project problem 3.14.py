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

# Define the inner and outer grid points
inner_points = np.zeros((N, N), dtype=bool)
inner_points[1:-1, 1:-1] = True
outer_points = ~inner_points

# Plot the domain and the boundary curves
fig, ax = plt.subplots(figsize=(6, 6))
ax.plot(gamma_1[:,0], gamma_1[:,1], 'b')
ax.plot(gamma_2[:,0], gamma_2[:,1], 'r')
ax.plot(gamma_3[:,0], gamma_3[:,1], 'g')

# Plot the inner and outer grid points
ax.plot(X[inner_points], Y[inner_points], 'bo', markersize=3)
ax.plot(X[outer_points], Y[outer_points], 'ro', markersize=3)

# Loop over all grid points to plot the normal lines to the fattened boundary
for i in range(N):
    for j in range(N):
        if (not inner_points[i, j]) or ((gamma_3[i, 0] - x_min)**2 + (gamma_3[i, 1] - y_min)**2 > d**2):
            continue
        nx = (gamma_3[i+1,1] - gamma_3[i-1,1])/(2*dx)
        ny = (gamma_3[i-1,0] - gamma_3[i+1,0])/(2*dy)
        norm = np.sqrt(nx**2 + ny**2)
        nx /= norm
        ny /= norm
        x_new = gamma_3[i,0] + d*nx
        y_new = gamma_3[i,1] + d*ny
        ax.plot([gamma_3[i,0], x_new], [gamma_3[i,1], y_new], 'r--')

# Set axis limits and labels
ax.set_xlim([x_min, x_max])
ax.set_ylim([y_min, y_max])
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.show()