import numpy as np
import matplotlib.pyplot as plt

# Define the domain
xmin, xmax, ymin, ymax = 0, 1, 0, 1
nx, ny = 51, 51
dx = (xmax - xmin) / (nx - 1)
dy = (ymax - ymin) / (ny - 1)

# Define the grid
x = np.linspace(xmin, xmax, nx)
y = np.linspace(ymin, ymax, ny)
X, Y = np.meshgrid(x, y)

# Define the boundary conditions
Tleft = np.zeros(ny)
Tright = np.zeros(ny)
Tbottom = np.zeros(nx)
Ttop = 1 - X**2

# Define the internal heat source
f = np.zeros((ny, nx))

# Set up the matrix A and vector b for the linear system Ax = b
A = np.zeros((nx*ny, nx*ny))
b = np.zeros(nx*ny)

# Fill the matrix A and vector b
for i in range(nx):
    for j in range(ny):
        k = i + j*nx # Index of the current grid point
        if i == 0 or i == nx-1 or j == 0 or j == ny-1: # If on the boundary
            A[k, k] = 1
            if i == 0:
                b[k] = Tleft[j]
            elif i == nx-1:
                b[k] = Tright[j]
            elif j == 0:
                b[k] = Tbottom[i]
            elif j == ny-1:
                # Set the values for all ny points at the top boundary
                b[i + np.arange(ny)*(nx)] = Ttop.flatten()
        else:
            A[k, k] = -2/dx**2 -2/dy**2
            A[k, k-1] = 1/dx**2
            A[k, k+1] = 1/dx**2
            A[k, k-nx] = 1/dy**2
            A[k, k+nx] = 1/dy**2
            b[k] = -f[j, i]

# Solve the linear system Ax = b
T = np.linalg.solve(A, b).reshape((ny, nx))

# Plot the solution
plt.contourf(X, Y, T, cmap='coolwarm')
plt.colorbar()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Poisson equation with Dirichlet boundary conditions')
plt.show()