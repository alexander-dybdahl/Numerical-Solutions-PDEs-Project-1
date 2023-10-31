import numpy as np
from scipy.sparse.linalg import spsolve
from scipy.sparse import lil_matrix, coo_matrix
import sympy as sp

def poisson_solver_2d(f, N):
    # Define the domain boundaries
    gamma_1 = lambda x: np.array([x, 0])
    gamma_2 = lambda y: np.array([0, y])
    gamma_3 = lambda x: np.array([x, 1 - x**2])

    # Define the number of interior points
    Nx = Ny = N
    h = 1 / (N+1)

    # Define the coordinates of the interior points
    x = np.linspace(h, 1 - h, Nx)
    y = np.linspace(h, 1 - h, Ny)

    # Create a meshgrid of the interior points
    X, Y = np.meshgrid(x, y, indexing='ij')

    # Initialize the solution vector
    u = np.zeros((Nx, Ny))

    # Construct the coefficient matrix A
    A = lil_matrix((Nx * Ny, Nx * Ny))
    for i in range(1, Nx - 1):
        for j in range(1, Ny - 1):
            A[i + j * Nx, i + j * Nx] = -4
            A[i + j * Nx, (i - 1) + j * Nx] = 1
            A[i + j * Nx, (i + 1) + j * Nx] = 1
            A[i + j * Nx, i + (j - 1) * Nx] = 1
            A[i + j * Nx, i + (j + 1) * Nx] = 1

    # Apply boundary conditions to the coefficient matrix A and the solution vector u
    for i in range(Nx):
        # Apply boundary condition gamma_1
        A[i, i] = 1
        u[i] = gamma_1(x[i])[1]

        # Apply boundary condition gamma_2
        A[i + (Ny - 1) * Nx, i + (Ny - 1) * Nx] = 1
        u[i + (Ny - 1) * Nx] = gamma_2(y[Ny - 1])[0]

    for j in range(1, Ny - 1):
        # Apply boundary condition gamma_3
        x_val = sp.symbols('x')
        eqn = (x_val - X[0, j])**2 + (1 - x_val**2 - Y[0, j])**2
        r = np.real(sp.solve(sp.diff(eqn, x_val), x_val)[0])
        i = int(round(r / h)) - 1
        A[i + j * Nx, :] = 0
        A[i + j * Nx, i + j * Nx] = 1
        u[i, j] = f(r, 1-r**2)

    # Convert the coefficient matrix A to a sparse matrix
    A = coo_matrix(A)

    # Solve the linear system
    u = u.flatten()
    u = spsolve(A, u)

    # Reshape the solution vector into a grid
    u = np.reshape(u, (Nx, Ny))

    return X, Y, u

# Test function
def f(x, y):
    return x**2-y**2

N = 100

T = poisson_solver_2d(f, N)