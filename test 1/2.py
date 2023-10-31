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
        if i == 0 or i == N-1 or j == 0 or j == N-1:
            A[k,k] = 1
            if i == 0 or i == N-1:
                b[k] = gamma_3[i,1]
            elif j == 0:
                b[k] = gamma_1[j,1]
            elif i != N-1:
                b[k] = gamma_2[i,0]
            else:
                b[k] = 0
        elif (gamma_3[i,0] - x_min)**2 + (gamma_3[i,1] - y_min)**2 <= d**2:
            # Extend the boundary value using normal extension
            xp, yp = j*dx, i*dy
            r = np.roots([2*d**3, 0, 1-2*yp, -xp])[0]
            x_new, y_new = r, 1-r**2
            A[k,k] = 1
            b[k] = T_new = T[int((y_new-y_min)/dy), int((x_new-x_min)/dx)]
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