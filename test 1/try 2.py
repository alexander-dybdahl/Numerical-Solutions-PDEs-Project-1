import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

N = 100 # number of grid points in each direction
h = 1 / (N - 1)

# Define the boundary curves
gamma_1 = np.array([(x, 0) for x in np.linspace(0, 1, N)])
gamma_2 = np.array([(0, y) for y in np.linspace(0, 1, N)])
gamma_3 = np.array([(x, 1 - x**2) for x in np.linspace(0, 1, N)])

# Define the fatten boundary distance
d = h/2

# Define the source term and Dirichlet boundary condition
f = lambda x, y: -np.pi**2 * np.sin(np.pi*x)
g = lambda x, y: np.sin(np.pi*x)

# Define the coefficient matrix and the right-hand side vector
A = np.zeros((N**2, N**2))
b = np.zeros(N**2)

# Normal extension
def normal_extension(xp, yp):
    roots = np.roots([1, yp-2*xp, xp-2*yp])
    r = np.real(roots[np.abs(roots - xp).argmin()])
    return r, 1-r**2

# Loop over all grid points to populate A and b
for i in range(N):
    for j in range(N):
        k = i*N + j
        x = i*h
        y = j*h
        if y == 1 - x**2:
            print('i')
        # Set the values on the boundary and in the fatten boundary region
        if i == 0 or i == N-1 or j == 0 or j == N-1:
            A[k,k] = 1
            b[k] = g(x, y)
        elif y > 1 - x**2 - d:
            A[k,k] = 1
            xq, yq = normal_extension(x, y)
            b[k] = g(x, y)
        # Set the values in the interior
        else:
            A[k,k] = -4/h**2
            A[k,k-1] = 1/h**2
            A[k,k+1] = 1/h**2
            A[k,k-N] = 1/h**2
            A[k,k+N] = 1/h**2
            b[k] = f(x, y)

# Solve the linear system using a direct solver
T = np.linalg.solve(A, b).reshape((N, N))

# Add the Dirichlet boundary values to the solution
for i in range(N):
    T[i,0] = g(i*h, 0)
    T[0,i] = g(0, i*h)
    T[i,-1] = g(i*h, 1-i*h**2)
    T[-1,:] = g(1, np.linspace(0, 1, N))

X, Y = np.meshgrid(np.linspace(0, 1, N), np.linspace(0, 1, N))
error = T.T - g(X, Y)

interior = np.zeros_like(X)
interior[Y <= 1 - X**2] = 1

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
ax = fig.add_subplot(projection='3d')
surf = ax.plot_surface(X, Y, g(X, Y), cmap=cm.coolwarm)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('T')
plt.title('Exact solution to ∆T = -f with fatten boundary approach')
plt.show()


# Plot the error
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
X, Y = np.meshgrid(np.linspace(0, 1, N), np.linspace(0, 1, N))
surf = ax.plot_surface(X, Y, error, cmap=cm.coolwarm)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('T')
plt.title('Error')
plt.show()