import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# Define the domain and grid parameters
x_min, x_max = 0, 1
y_min, y_max = 0, 1
N = 100 # number of grid points in each direction
dx = (x_max - x_min) / (N - 1)
dy = (y_max - y_min) / (N - 1)

# Define the boundary curves
gamma_1 = np.array([(x, 0) for x in np.linspace(x_min, x_max, N)])
gamma_2 = np.array([(0, y) for y in np.linspace(y_min, y_max, N)])
gamma_3 = np.array([(x, 1 - x**2) for x in np.linspace(x_min, x_max, N)])

# Define the fatten boundary distance
d = dx/2

# Define the source term and Dirichlet boundary condition
f = lambda x, y: -np.pi**2 * np.sin(np.pi*x)
g = lambda x, y: np.sin(np.pi*x)

# Define the coefficient matrix and the right-hand side vector
A = np.zeros((N**2, N**2))
b = np.zeros(N**2)

# Loop over all grid points to populate A and b
for i in range(N):
    for j in range(N):
        k = i*N + j
        # Set the values on the boundary and in the fatten boundary region
        if i == 0 or i == N-1 or j == 0 or j == N-1 or (i*dx)**2 + (j*dy-1)**2 <= d**2:
            A[k,k] = 1
            b[k] = g(i*dx, j*dy)
        # Set the values in the interior
        else:
            A[k,k] = -2/dx**2 - 2/dy**2
            A[k,k-1] = 1/dx**2
            A[k,k+1] = 1/dx**2
            A[k,k-N] = 1/dy**2
            A[k,k+N] = 1/dy**2
            b[k] = f(i*dx, j*dy)

# Solve the linear system using a direct solver
T = np.linalg.solve(A, b).reshape((N, N))

# Add the Dirichlet boundary values to the solution
for i in range(N):
    T[i,0] = g(i*dx, 0)
    T[0,i] = g(0, i*dy)
    T[i,-1] = g(i*dx, 1)
    T[-1,:] = g(1, np.linspace(y_min, y_max, N))

X, Y = np.meshgrid(np.linspace(x_min, x_max, N), np.linspace(y_min, y_max, N))

interior = np.zeros_like(X)
interior[Y < 1 - X**2] = 1

T[interior == 0] = None
X[interior == 0] = None
Y[interior == 0] = None

# Plot the solution
ax = plt.subplot(projection="3d")
ax.plot_surface(X, Y, T.T, cmap=cm.coolwarm)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('T')
plt.title('Solution to ∆T = -f with fatten boundary approach')
plt.show()

# Plot the exact solution
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, Y, g(X, Y), cmap=cm.coolwarm)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('T')
plt.title('Exact solution to ∆T = -f with fatten boundary approach')
plt.show()


# Plot the error
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, Y, T.T - g(X, Y), cmap=cm.coolwarm)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('T')
plt.title('Error')
plt.show()