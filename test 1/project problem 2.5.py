import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

# Define the domain
Lx, Ly = 1, 1
Nx, Ny = 51, 51
dx, dy = Lx/(Nx-1), Ly/(Ny-1)
x = np.linspace(0, Lx, Nx)
y = np.linspace(0, Ly, Ny)
X, Y = np.meshgrid(x, y)

# Define the thickness of the fatten boundary region
D = dx/2

# Define the diffusion coefficient
k = 1

# Define the boundary conditions
T1 = np.zeros(Ny)
T2 = np.zeros(Nx)
T3 = 1 - X**2

# Initialize the coefficient matrix and the right-hand side vector
A = np.zeros((Nx*Ny, Nx*Ny))
b = np.zeros(Nx*Ny)

# Fill in the coefficient matrix and the right-hand side vector
for i in range(Nx):
    for j in range(Ny):
        idx = j*Nx + i
        
        # Check if the point is on the boundary
        if i == 0 or j == 0 or i == Nx-1 or j == Ny-1 or (i*dx)**2 + (j*dy-1)**2 <= D**2:
            A[idx, idx] = 1
            b[idx] = T1[j] if i == 0 else (T2[i] if j == 0 else T3[i])
        else:
            # Compute the coefficients for the discretization of the Laplacian
            ax = k/(dx**2)
            ay = k/(dy**2)
            bx = -2*k/(dx**2) - 2*k/(dy**2)
            if i > 0:
                A[idx, idx-1] = ay
                bx -= ay
            if j > 0:
                A[idx, idx-Nx] = ax
                bx -= ax
            if i < Nx-1:
                A[idx, idx+1] = ay
                bx -= ay
            if j < Ny-1:
                A[idx, idx+Nx] = ax
                bx -= ax
            A[idx, idx] = bx

# Solve the linear system using sparse matrix solver
T = spsolve(diags(np.diag(A)), b)

# Reshape the solution into a 2D array
T = T.reshape((Ny, Nx))

# Compute the exact solution for comparison
Texact = X*(1-X)*Y*(1-Y)

# Compute the error
error = np.max(np.abs(T - Texact))

# Print the maximum error
print("Maximum error: {:.6f}".format(error))

# Plot the solution
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, T, cmap='coolwarm')
ax.set_xlabel('x')
ax.set_xlabel('y')
plt.show()