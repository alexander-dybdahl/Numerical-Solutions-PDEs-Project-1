import numpy as np
import matplotlib.pyplot as plt

# Define the domain and boundary
x_min, x_max = 0, 1
y_min, y_max = 0, 1
nx, ny = 50, 50
dx = (x_max - x_min) / (nx - 1)
dy = (y_max - y_min) / (ny - 1)
x = np.linspace(x_min, x_max, nx)
y = np.linspace(y_min, y_max, ny)
X, Y = np.meshgrid(x, y)
gamma_1 = np.array([(i, 0) for i in x])
gamma_2 = np.array([(0, j) for j in y])
gamma_3 = np.array([(i, 1 - i**2) for i in x])

# Define the exact solution
def exact_solution(x, y):
    return np.sin(np.pi * x) * np.sin(np.pi * y)

# Define the diffusion coefficient
def kappa(x, y):
    return np.ones_like(x)

# Define the right-hand side of the Poisson equation
def f(x, y):
    return -2 * np.pi**2 * exact_solution(x, y)

# Define the Dirichlet boundary conditions
def g(x, y):
    return exact_solution(x, y)

# Construct the coefficient matrix
A = np.zeros((nx*ny, nx*ny))
for i in range(nx):
    for j in range(ny):
        k = j * nx + i
        A[k, k] = dx**2 * dy**2 * kappa(x[i], y[j])
        if i > 0:
            A[k, k-1] = -0.5 * dx**2 * dy**2 * kappa(x[i-1], y[j])
        if i < nx-1:
            A[k, k+1] = -0.5 * dx**2 * dy**2 * kappa(x[i+1], y[j])
        if j > 0:
            A[k, k-nx] = -0.5 * dx**2 * dy**2 * kappa(x[i], y[j-1])
        if j < ny-1:
            A[k, k+nx] = -0.5 * dx**2 * dy**2 * kappa(x[i], y[j+1])

# Construct the right-hand side vector
rhs = np.zeros(nx*ny)
for i in range(nx):
    for j in range(ny):
        k = j * nx + i
        rhs[k] = dx**2 * dy**2 * f(x[i], y[j])
        if i == 0:
            rhs[k] += g(x[i], y[j-1]) * dy**2
        if i == nx-1:
            rhs[k] += g(x[i], y[j-1]) * dy**2
        if j == 0:
            rhs[k] += g(x[i], y[j]) * dx**2
        if j == ny-1:
            rhs[k] += g(x[i], y[j]) * dx**2

# Solve the linear system
T = np.linalg.solve(A, rhs)

# Compute the exact solution and error
T_exact = exact_solution(X, Y)
error = T_exact - T.reshape((ny, nx))

# Plot the results
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].pcolormesh(X, Y, T.reshape((ny, ny)))
plt.show()