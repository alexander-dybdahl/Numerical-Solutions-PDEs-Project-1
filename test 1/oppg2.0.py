import numpy as np
import matplotlib.pyplot as plt

# Define the function to solve
def f(x, y):
    return np.sin(np.pi * x) * np.sin(np.pi * y)

# Define the domain and grid size
a, b = 0, 1
nx, ny = 20, 20
h = (b-a) / (nx-1)

# Generate the grid
x = np.linspace(a, b, nx)
y = np.linspace(a, b, ny)
X, Y = np.meshgrid(x, y)

# Compute the distance to the boundary
dist = np.minimum(np.minimum(X-a, b-X), np.minimum(Y-a, b-Y))
boundary = dist < h/2

# Compute the distance to the fattened boundary
dist_fat = dist - h/2

# Identify boundary points and the north and east boundaries
boundary_hit = boundary.astype(int)
north_boundary = np.zeros_like(X)
east_boundary = np.zeros_like(X)
for i in range(nx):
    for j in range(ny):
        if boundary[i,j]:
            if j < ny-1 and not boundary[i,j+1]:
                north_boundary[i,j] = 1
            if i < nx-1 and not boundary[i+1,j]:
                east_boundary[i,j] = 1

# Compute the solution using the finite difference method
interior_no_bnd = np.zeros_like(X)
for k in range(10000):
    P = boundary_hit + (4*interior_no_bnd + north_boundary*np.divide(2, dist_fat[:,1:], out=np.zeros_like(X), where=dist_fat[:,1:] != 0) + east_boundary*np.divide(2, dist_fat[1:,:], out=np.zeros_like(X), where=dist_fat[1:,:] != 0)) / h**2
    interior_no_bnd = np.maximum(0, (P - north_boundary*np.divide(2, 1+dist_fat[:,:-1], out=np.zeros_like(X), where=dist_fat[:,:-1] != 0))[:,:-1] + P[:,:-1] - (P[:,1:] + north_boundary*np.divide(2, 1+dist_fat[:,1:], out=np.zeros_like(X), where=dist_fat[:,1:] != 0))[:,1:] + P[:-1,:] - east_boundary*np.divide(2, 1+dist_fat[:-1,:], out=np.zeros_like(X), where=dist_fat[:-1,:] != 0) - P[1:,:] - east_boundary*np.divide(2, 1+dist_fat[1:,:], out=np.zeros_like(X), where=dist_fat[1:,:] != 0))
    
# Plot the solution
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, interior_no_bnd, cmap='viridis')
ax.set_title("Numerical solution")
plt.show()