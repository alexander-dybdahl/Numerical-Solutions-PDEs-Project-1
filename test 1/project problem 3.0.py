import numpy as np
import matplotlib.pyplot as plt

# Define the domain and boundary
x0, x1, y0, y1 = 0, 1, 0, 1
nx, ny = 101, 101
x = np.linspace(x0, x1, nx)
y = np.linspace(y0, y1, ny)
X, Y = np.meshgrid(x, y)
f = np.zeros((nx, ny))
T = np.zeros((nx, ny))
T[0, :] = 0     # Dirichlet boundary condition on gamma_1
T[:, 0] = 0     # Dirichlet boundary condition on gamma_2
T[-1, :] = 0    # Dirichlet boundary condition on gamma_3

# Define the grid spacings
dx = x[1] - x[0]
dy = y[1] - y[0]

# Define the iteration parameters
tol = 1e-6      # tolerance for convergence
max_iter = 1000 # maximum number of iterations

# Define the width of the fattened boundary
d = 5*dx

# Define the interior points
interior_points = (X > d) & (X < (x1-d)) & (Y > d) & (Y < (y1-d))

# Iterate to solve the Poisson equation
for k in range(max_iter):
    T0 = T.copy()
    for i in range(1, nx-1):
        for j in range(1, ny-1):
            if not interior_points[i,j]:    # handle fattened boundary
                T[i, j] = (T[i-1, j] + T[i+1, j] + T[i, j-1] + T[i, j+1]) / 4
            else:   # handle interior points
                T[i, j] = (f[i, j] - (T[i+1, j] + T[i-1, j] - 2*T[i, j]) / dx**2 - (T[i, j+1] + T[i, j-1] - 2*T[i, j]) / dy**2) / (-2*(1/dx**2 + 1/dy**2))
    # Check for convergence
    if np.max(np.abs(T - T0)) < tol:
        break

# Plot the solution
vmin = T.min()
vmax = T.max()
levels = np.linspace(vmin, vmax, 21)
plt.imshow(T.T, origin='lower', extent=[x0, x1, y0, y1], cmap='jet')
plt.colorbar()
plt.contour(X, Y, T.T, colors='w', linestyles='-', levels=levels)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Poisson equation solution with Dirichlet boundary conditions (using fattened boundary)')
plt.show()