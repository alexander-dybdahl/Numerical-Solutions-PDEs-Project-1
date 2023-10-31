import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the domain and boundary
x_min, x_max = 0, 1
y_min, y_max = 0, 1
N = 100  # Number of interior grid points in each direction
dx = (x_max - x_min) / (N + 1)
dy = (y_max - y_min) / (N + 1)

x = np.linspace(x_min, x_max, N + 2)
y = np.linspace(y_min, y_max, N + 2)

X, Y = np.meshgrid(x, y)

# Define the function kappa(x,y) = I
def kappa(x, y):
    return np.eye(2)

# Define the function f(x,y) = 1
def f(x, y):
    return np.ones_like(x)

# Define the boundary conditions
g1 = np.zeros(N+2)
g2 = np.zeros(N+2)
g3 = 1 - x**2

# Create the coefficient matrix
A = np.zeros((N**2, N**2))
b = np.zeros(N**2)

for i in range(N):
    for j in range(N):
        n = i * N + j
        A[n, n] = -2 * (dx**2 + dy**2)
        if i == 0:
            A[n, n + N] = dy**2
            b[n] -= g1[j+1] * dy**2
        elif i == N - 1:
            A[n, n - N] = dy**2
            b[n] -= g2[j+1] * dy**2
        else:
            A[n, n + N] = dy**2 / (1 + dy / dx)**2
            A[n, n - N] = dy**2 / (1 + dy / dx)**2
        
        if j == 0:
            A[n, n + 1] = dx**2
            b[n] -= g3[i+1] * dx**2 / (1 + dx / dy)**2
        elif j == N - 1:
            A[n, n - 1] = dx**2
            b[n] -= g3[i+1] * dx**2 / (1 + dx / dy)**2
        else:
            A[n, n + 1] = dx**2 / (1 + dx / dy)**2
            A[n, n - 1] = dx**2 / (1 + dx / dy)**2

# Solve the system of equations
b = dx**2 * dy**2 * f(X[1:-1, 1:-1], Y[1:-1, 1:-1]).flatten()
T = np.linalg.solve(A, b).reshape((N, N))

# Plot the solution
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X[1:-1, 1:-1], Y[1:-1, 1:-1], T, cmap='coolwarm')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('T')
plt.show()