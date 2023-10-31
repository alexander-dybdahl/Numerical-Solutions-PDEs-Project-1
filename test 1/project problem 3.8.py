import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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

# Define the source term
f = np.zeros((N, N))

# Define the coefficient matrix and the right-hand side vector
A = np.zeros((N**2, N**2))
b = np.zeros(N**2)

# Loop over all grid points to populate A and b
for i in range(N):
    for j in range(N):
        k = i*N + j
        # Set the values on the boundary and in the fatten boundary region
        if i == 0 or i == N-1 or j == 0 or j == N-1 or ((i*dx-gamma_3[j,0])**2 + (j*dy-1+gamma_3[j,1])**2) <= d**2:
            A[k,k] = 1
            if i == 0 or i == N-1:
                b[k] = gamma_3[j,1]
            elif j == 0:
                b[k] = gamma_1[i,1]
            elif i == N-1:
                b[k] = 0
            else:
                xp = i*dx
                yp = j*dy
                coeffs = [-2, 0, yp-1+2*xp, -2*xp*yp + xp - yp]
                r_roots = np.roots(coeffs)
                r = r_roots[np.where(np.isreal(r_roots) & (r_roots > 0))][0]
                xQ = np.array([r, 1-r**2])
                n = np.array([2*r**3 - r + xp, 1 - 2*r**2 - yp])
                n = n / np.linalg.norm(n)
                b[k] = np.dot(n, gamma_3[j,:] - xQ) + T[i-1,j] * np.dot(n, np.array([1, 0])) / dx**2 + T[i+1,j] * np.dot(n, np.array([-1, 0])) / dx**2 + T[i,j-1] * np.dot(n, np.array([0, 1])) / dy**2 + T[i,j+1] * np.dot(n, np.array([0, -1])) / dy**2
        # Set the values in the interior
        else:
            A[k,k] = -2/dx**2 - 2/dy**2
            A[k,k-1] = 1/dx**2
            A[k,k+1] = 1/dx**2
            A[k,k-N] = 1/dy**2
            A[k,k+N] = 1/dy**2
            b[k] = f[i,j]

# Solve the linear system using a direct solver
T = np.linalg.solve(A, b).reshape((N, N))

# Plot the solution as a surface plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
X, Y = np.meshgrid(np.linspace(x_min, x_max, N), np.linspace(y_min, y_max, N))
surf = ax.plot_surface(X, Y, T.T, cmap='jet', linewidth=0, antialiased=False)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('T')
plt.title('Solution to ∇ · (κ∇T) = ∆T with fatten boundary approach')
plt.show()