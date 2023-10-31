import numpy as np
import matplotlib.pyplot as plt

# Define the domain
x_min, x_max = 0, 1
y_min, y_max = 0, 1
nx, ny = 101, 101
dx = (x_max - x_min) / (nx - 1)
dy = (y_max - y_min) / (ny - 1)

# Define the boundary
def boundary(x, y):
    if y == 0:
        return 0
    elif x == 0:
        return y
    elif x == 1:
        return 1 - y**2
    else:
        return 0

# Define the function f(x, y) = ∆T
def source(x, y):
    return 0

# Define the coefficient function κ(x, y) = 1
def kappa(x, y):
    return 1

# Set up the grid
x = np.linspace(x_min, x_max, nx)
y = np.linspace(y_min, y_max, ny)
X, Y = np.meshgrid(x, y)

# Fatten the boundary by adding ghost points
T = np.zeros((nx+2, ny+2))
T[1:-1,1:-1] = np.array([[boundary(xi, yj) for xi in x] for yj in y])
T[0,:] = T[1,:]
T[-1,:] = T[-2,:]
T[:,0] = T[:,1]
T[:,-1] = T[:,-2]

# Set up the coefficient matrix
A = np.zeros((nx*ny, nx*ny))
for i in range(nx):
    for j in range(ny):
        k = i + j*nx
        A[k,k] = -2*(dx**2 + dy**2)/kappa(x[i], y[j])
        if i > 0:
            A[k,k-1] = dx**2/kappa(x[i], y[j])
        if i < nx-1:
            A[k,k+1] = dx**2/kappa(x[i], y[j])
        if j > 0:
            A[k,k-nx] = dy**2/kappa(x[i], y[j])
        if j < ny-1:
            A[k,k+nx] = dy**2/kappa(x[i], y[j])

# Set up the right-hand side vector
b = np.zeros(nx*ny)
for i in range(nx):
    for j in range(ny):
        k = i + j*nx
        b[k] = source(x[i], y[j])*dx**2*dy**2/kappa(x[i], y[j])

# Solve the linear system
T_fatten = np.linalg.solve(A, b)

# Reshape the solution and remove the ghost points
T_fatten = T_fatten.reshape((nx, ny))
T_fatten = T_fatten[1:-1,1:-1]

# Plot the solution
fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.set_title('Solution using fatten the boundary')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.contourf(X, Y, T_fatten, levels=50)
plt.show()