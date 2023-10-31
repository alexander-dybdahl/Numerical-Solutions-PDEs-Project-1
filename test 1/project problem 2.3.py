import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the domain
nx = ny = 101
Lx = Ly = 1.0
x = np.linspace(0, Lx, nx)
y = np.linspace(0, Ly, ny)
X, Y = np.meshgrid(x, y)
dx = x[1] - x[0]
dy = y[1] - y[0]

def create_coefficient_matrix(shape):
    ny, nx = shape
    h = 1 / (nx - 1)
    main_diag = np.ones(nx*ny) * (-4/h**2)
    off_diag = np.ones((nx-1)*ny) * (1/h**2)
    off_diag[nx-1::nx] = 0
    diagonals = [main_diag, off_diag, off_diag, off_diag, off_diag]
    offsets = [0, -1, 1, -nx, nx]
    A = np.diag(diagonals[0]) + np.diag(diagonals[1], offsets[1]) + np.diag(diagonals[2], offsets[2]) + np.diag(diagonals[3], offsets[3]) + np.diag(diagonals[4], offsets[4])
    return A

nx = ny = 101
A = create_coefficient_matrix((101, 101))

# Define the boundary conditions
T = np.zeros((ny, nx))
T[0, :] = 0  # T = 0 on gamma_1
T[:, 0] = 0  # T = 0 on gamma_2
T[-1, :] = 1 - X[-1, :]**2  # T = 1 - x^2 on gamma_3

# Fatten the boundary
T[1:-1, 1:-1] = np.nan
for i in range(1, nx-1):
    for j in range(1, ny-1):
        if np.isnan(T[j, i]):
            T[j, i] = (T[j-1, i] + T[j+1, i] + T[j, i-1] + T[j, i+1]) / 4

# Solve the system of equations
b = -np.ones(nx*ny) * (1/dx**2 + 1/dy**2)
b[0:nx] = 0  # T = 0 on gamma_1
b[::nx] = 0  # T = 0 on gamma_2
b[-nx:] = (1/dx**2) * (1 - X[-1, :]**2)  # T = 1 - x^2 on gamma_3
T_flat = np.reshape(T, nx*ny)
T_flat = np.linalg.solve(A, b)
T = np.reshape(T_flat, (ny, nx))

# Plot the solution
fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X, Y, T, cmap=plt.cm.coolwarm)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('T')
plt.show()