import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt

# Define the domain
xmin, xmax, ymin, ymax = 0, 1, 0, 1
nx, ny = 101, 101
dx, dy = (xmax-xmin)/(nx-1), (ymax-ymin)/(ny-1)
x = np.linspace(xmin, xmax, nx)
y = np.linspace(ymin, ymax, ny)
X, Y = np.meshgrid(x, y)

# Define the thickness of the fatten boundary region
D = dx/2

# Define the boundary condition function
def bc(x, y):
    if x == 0:
        return y
    elif y == 0:
        return 2*x
    else:
        r = np.roots([2, -2*y, 1, -x])[0].real
        d = np.sqrt((x-r)**2 + (y-(1-r**2))**2)
        if d <= D:
            n = np.array([-2*r, 1-2*r**2])
            n /= np.linalg.norm(n)
            tr = r
            T = tr + n[0]*(x-r) + n[1]*(y-(1-r**2))
        else:
            T = None
        return T

# Define the source function
def f(x, y):
    return 2*(x-y)

# Construct the coefficient matrix and the right-hand side vector
A = np.zeros((nx*ny, nx*ny))
b = np.zeros(nx*ny)
for i in range(nx):
    for j in range(ny):
        k = i + j*nx
        if i == 0 or i == nx-1 or j == 0 or j == ny-1:
            A[k,k] = 1
            b[k] = bc(x[i], y[j])
        else:
            A[k,k] = -2/dx**2 - 2/dy**2
            A[k,k-1] = 1/dx**2
            A[k,k+1] = 1/dx**2
            A[k,k-nx] = 1/dy**2
            A[k,k+nx] = 1/dy**2
            b[k] = f(x[i], y[j])

# Solve the linear system
T = spsolve(diags(A), b).reshape(ny, nx)

# Compute the analytical solution for comparison
T_analytical = X**2 - Y**2

# Compute the maximum error
error = np.max(np.abs(T - T_analytical))

# Plot the numerical and analytical solutions
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10,4))
ax1.set_title('Numerical solution')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_xlim(xmin, xmax)
ax1.set_ylim(ymin, ymax)
c = ax1.contourf(X, Y, T, cmap='coolwarm')
plt.colorbar(c, ax=ax1)
ax2.set_title('Analytical solution')
ax2.set_xlabel('x')
ax2.set_xlim(xmin, xmax)
ax2.set_ylim(ymin, ymax)
c = ax2.contourf(X, Y, T_analytical, cmap='coolwarm')
plt.colorbar(c, ax=ax2)
plt.show()

# Plot