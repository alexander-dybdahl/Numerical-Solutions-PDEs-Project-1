import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import spdiags, eye
from scipy.sparse.linalg import spsolve

def solve_poisson_pde(f, kappa, nx, ny, dx, dy, fat_width):
    # Define the number of interior grid points
    nx_int = nx - 2*fat_width
    ny_int = ny - 2*fat_width
    
    # Define the stencil coefficients
    main_diag = -2*np.ones(nx_int*ny_int)
    off_diag = np.ones(nx_int*ny_int-1)
    off_diag[nx_int-1::nx_int] = 0
    off_diag[::nx_int] = 0
    diag_offsets = [-nx_int, -1, 0, 1, nx_int]
    stencils = [main_diag, off_diag, off_diag, off_diag, off_diag]
    A = spdiags(stencils, diag_offsets, nx_int*ny_int, nx_int*ny_int)
    
    # Define the domain grid
    x = np.linspace(dx/2, (nx-0.5)*dx, nx)
    y = np.linspace(dy/2, (ny-0.5)*dy, ny)
    X, Y = np.meshgrid(x, y)
    
    # Define the source term
    F = f(X,Y)
    F = F[fat_width:-fat_width, fat_width:-fat_width].flatten()
    
    # Define the diffusion coefficient
    kappa = kappa(X,Y)
    kappa = kappa[fat_width:-fat_width, fat_width:-fat_width].flatten()
    
    # Define the boundary conditions
    bc_values = np.zeros(nx*ny)
    bc_indices = np.arange(nx*fat_width)
    bc_values[bc_indices] = fatten_boundary(bc_indices//nx, nx_int, ny_int, fat_width)
    bc_indices = np.arange(nx*ny-nx*fat_width, nx*ny)
    bc_values[bc_indices] = fatten_boundary(bc_indices//nx, nx_int, ny_int, fat_width)
    bc_indices = np.arange(fat_width*nx)
    bc_values[bc_indices[::nx]] = fatten_boundary(bc_indices[:nx], nx_int, ny_int, fat_width)
    bc_indices = np.arange(fat_width*nx, nx*ny-fat_width*nx, nx)
    bc_values[bc_indices] = fatten_boundary(bc_indices//nx-fat_width, nx_int, ny_int, fat_width)
    A = A.tolil()
    A[bc_indices, :] = 0
    A[bc_indices, bc_indices] = 1
    F -= A.dot(bc_values)
    
    # Solve the linear system
    T = spsolve(A.tocsc(), F)
    T = np.pad(T.reshape((ny_int, nx_int)), ((fat_width, fat_width), (fat_width, fat_width)), 'constant')
    
    return X, Y, T

def fatten_boundary(indices, nx_int, ny_int, fat_width):
    return np.linspace(0, 1, nx_int+2*fat_width)[1:-1]**2

# Define the problem parameters
nx = 101
ny = 101
dx = 0.1
dy = 0.1
kappa = lambda x,y: np.ones_like(x)
f = lambda x,y: np.zeros_like(x)

# Solve the PDE
fat_width = 10
X, Y, T = solve_poisson_pde(f, kappa, nx, ny, dx, dy, fat_width)