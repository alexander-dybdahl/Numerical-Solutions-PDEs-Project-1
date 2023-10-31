import numpy as np
import matplotlib.pyplot as plt

# Define domain and grid parameters
Lx, Ly = 1, 1
nx, ny = 101, 101
dx, dy = Lx/(nx-1), Ly/(ny-1)
x = np.linspace(0, Lx, nx)
y = np.linspace(0, Ly, ny)
X, Y = np.meshgrid(x, y)

# Define boundary conditions
T_left = np.zeros(ny)
T_right = np.zeros(ny)
T_top = 1 - X**2
T_bottom = np.zeros(nx)

# Fatten boundary using normal extension method
D = dx/2  # Thickness of fatten boundary region
T = np.zeros((ny, nx))
for i in range(nx):
    for j in range(ny):
        # Compute distance from (x,y) to curve gamma3
        xP, yP = x[i], y[j]
        r = np.roots([2, 1-2*yP, -xP])[0]
        xQ, yQ = r, 1 - r**2
        d = np.sqrt((xP - xQ)**2 + (yP - yQ)**2)
        if d <= D:
            # Compute normal vector to gamma3 at (xQ,yQ)
            n = np.array([-2*r, 1-2*r**2])
            n = n/np.linalg.norm(n)
            # Extend boundary condition to (x,y) using normal extension formula
            T_boundary = T_top[i] if j == ny-1 else \
                        T_bottom[i] if j == 0 else \
                        T_left[j] if i == 0 else \
                        T_right[j] if i == nx-1 else 0
            T[j,i] = T_boundary + np.dot(np.array([xP,yP])-np.array([xQ,yQ]), n)*np.gradient(T_top)[0]

# Assemble coefficient matrix and right-hand side vector
A = np.zeros((nx*ny, nx*ny))
b = np.zeros(nx*ny)
for i in range(nx):
    for j in range(ny):
        n = j*nx + i
        A[n,n] = -2/dx**2 - 2/dy**2
        if i > 0:
            A[n,n-1] = 1/dx**2
        if i < nx-1:
            A[n,n+1] = 1/dx**2
        if j > 0:
            A[n,n-nx] = 1/dy**2
        if j < ny-1:
            A[n,n+nx] = 1/dy**2
        if i == 0:
            b[n] -= T_left[j]/dx**2
        if i == nx-1:
            b[n] -= T_right[j]/dx**2
        if j == 0:
            b[n] -= T_bottom[i]/dy**2
        if j == ny-1:
            b[n] -= T_top[i]/dy**2

# Solve linear system of equations
T_flat = np.linalg.solve(A,b)
T = T_flat.reshape((ny, nx))

# Compute exact solution for comparison
T_exact = 0.5*X**2 - 0.25*Y**4 - 0.25