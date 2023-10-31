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

# Define the finite difference coefficients
a = 1 / dx**2
b = 1 / dy**2
c = -2 * (a + b)

# Define the iteration parameters
tol = 1e-6      # tolerance for convergence
max_iter = 1000 # maximum number of iterations

# Iterate to solve the Poisson equation
for k in range(max_iter):
    T0 = T.copy()
    for i in range(1, nx-1):
        for j in range(1, ny-1):
            if (i == 1):    # handle gamma_1 boundary
                T[i-1, j] = 2 * T[i, j] - T[i+1, j]
            elif (i == nx-2):  # handle gamma_3 boundary
                T[i+1, j] = (T[i, j] + dx * f[i, j]) / (1 + dx)
            if (j == 1):    # handle gamma_2 boundary
                T[i, j-1] = 2 * T[i, j] - T[i, j+1]
            elif (j == ny-2):
                T[i, j+1] = (T[i, j] + dy * f[i, j]) / (1 + dy)
            else:   # handle interior points
                T[i, j] = (f[i, j] - a * (T[i+1, j] + T[i-1, j]) - b * (T[i, j+1] + T[i, j-1])) / c
    # Check for convergence
    if np.max(np.abs(T - T0)) < tol:
        break

# Plot the solution
plt.imshow(T.T, origin='lower', extent=[x0, x1, y0, y1], cmap='jet')
plt.colorbar()
plt.contour(X, Y, T.T, colors='w', linestyles='-', levels=10)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Poisson equation solution with Dirichlet boundary conditions')
plt.show()