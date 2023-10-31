import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import bisect

# Define the domain and grid parameters
Lx, Ly = 1.0, 1.0
Nx, Ny = 101, 101
dx, dy = Lx/(Nx-1), Ly/(Ny-1)
x = np.linspace(0, Lx, Nx)
y = np.linspace(0, Ly, Ny)
X, Y = np.meshgrid(x, y)

# Define the fatten boundary parameters
D = dx/2  # thickness of the fatten boundary
gamma3 = lambda x: 1 - x**2  # curve defining the boundary

# Define the boundary condition function
def g(x, y):
    return np.sin(np.pi*y/Ly)

# Construct the coefficient matrix and right-hand side vector
A = np.zeros((Nx*Ny, Nx*Ny))
b = np.zeros(Nx*Ny)

for i in range(Nx):
    for j in range(Ny):
        p = i*Ny + j  # index of the current grid point
        x_p, y_p = i*dx, j*dy  # Cartesian coordinates of the current grid point
        
        # Check if the current point is within the fatten boundary
        r = bisect(lambda r: x_p + r*(1-2*y_p) - 2*r**3, 0, 1)
        d = np.sqrt((x_p - r)**2 + (y_p - (1-r**2))**2)
        if d <= D:
            # Compute the normal vector to the curve gamma3 at (r, 1-r^2)
            n = np.array([-2*r, 1-2*r**2])
            
            # Compute the extended boundary condition at (x_p, y_p)
            T_r = g(r, 1-r**2)
            grad_T_r = np.array([np.pi/Ly*np.cos(np.pi*(1-r**2)/Ly)*2*r, -2*np.pi*r/Ly*np.sin(np.pi*(1-r**2)/Ly)])
            T_p = T_r + np.dot(grad_T_r, [x_p-r, y_p-(1-r**2)])
            b[p] = T_p
            A[p, p] = 1
        else:
            # Set the standard boundary condition at (x_p, y_p)
            b[p] = g(x_p, y_p)
            if i > 0:
                q = (i-1)*Ny + j
                A[p, q] = 1/(dx**2)
            if i < Nx-1:
                q = (i+1)*Ny + j
                A[p, q] = 1/(dx**2)
            if j > 0:
                q = i*Ny + (j-1)
                A[p, q] = 1/(dy**2)
            if j < Ny-1:
                q = i*Ny + (j+1)
                A[p, q] = 1/(dy**2)

# Solve the linear system using the conjugate gradient method
T = np.linalg.solve(A, b)

# Reshape the solution as a 2D array and plot it as a surface
T = T.reshape((Nx, Ny))
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, T)
plt.show