import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

# Define domain and grid
N = 100  # Number of grid points in each direction
x = np.linspace(0, 1, N)
y = np.linspace(0, 1, N)
X, Y = np.meshgrid(x, y)
dx = x[1] - x[0]
dy = y[1] - y[0]

# Define boundary
gamma1 = np.array([[x, 0] for x in np.linspace(0, 1, N)])
gamma2 = np.array([[0, y] for y in np.linspace(0, 1, N)])
gamma3 = np.array([[x, 1 - x**2] for x in np.linspace(0, 1, N)])
boundary = np.vstack((gamma1, gamma2, gamma3))

# Fatten boundary
def fatten_boundary(x, y, thickness):
    r = np.zeros_like(x)
    for i in range(len(x)):
        # Find intersection of normal line and gamma3
        def f(r):
            return x[i] + r*(1 - 2*y[i]) - 2*r**3
        r[i] = fsolve(f, x0=y[i])[0]
    # Compute distance to boundary
    d = np.sqrt((x - r)**2 + (y - 1 + r**2)**2)
    return np.where(d < thickness, np.nan, 0)

thickness = 0.05
F = fatten_boundary(X, Y, thickness)

# Define diffusion coefficient
kappa = np.ones_like(X)
D = np.ones_like(X) / (dx**2)

# Define right-hand side
f = np.zeros_like(X)

# Define solution matrix
T = np.zeros_like(X)

# Set boundary conditions
T[0,:] = 1
T[:,0] = 0
T[:,-1] = 0
T[np.isnan(F)] = 0

# Define Laplacian matrix
diags_main = [-2*np.ones(N), np.ones(N-1), np.ones(N-1)]
L = diags(diags_main, [0, -1, 1], shape=(N, N)) / dx**2

# Compute solution
T_flat = T.flatten()
F_flat = F.flatten()
for i in range(1000):
    T_flat = spsolve(-L + diags(F_flat*D), f.flatten())
    T = T_flat.reshape((N, N))
    T[0,:] = 1
    T[:,0] = 0
    T[:,-1] = 0
    T[np.isnan(F)] = 0
    F_flat = fatten_boundary(X.flatten(), Y.flatten(), thickness).flatten()

# Compute exact solution for comparison
def exact_solution(x, y):
    return x*(1 - x**2)*y*(1 - y)
exact = exact_solution(X, Y)

# Compute error and plot convergence
errors = []
for i in range(5, 100, 5):
    xi = np.linspace(0, 1, i)
    yi = np.linspace(0, 1, i)
    Xi, Yi = np.meshgrid(xi, yi)
    dx = xi[1] - xi[0]
    dy = yi[1] - yi[0]
    Ti = exact_solution(Xi, Yi)
    Fi = fatten_boundary(Xi, Yi, thickness)
    Di = np.ones_like(Xi) / (