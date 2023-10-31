import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

N = 100 # number of grid points in each direction

# Define the function f on the boundary curve gamma3
def f(x, y):
    return x**2-y**2

def poisson(f, N):
    # Define the domain and boundary curves
    x = np.linspace(0, 1, N+1) # grid points in x direction
    y = np.linspace(0, 1, N+1) # grid points in y direction
    X, Y = np.meshgrid(x, y) # 2D grid of points
    h = 1/N

    gamma1 = np.zeros(N-1,) # boundary condition on gamma1
    gamma2 = np.zeros(N-1,) # boundary condition on gamma2
    gamma3 = 1 - x**2 # boundary condition on gamma3
    
    # Construct the right-hand side (heat source) function
    H = np.zeros((N-1, N-1)) # zero heat source inside the domain
    
    for i in range(1, N):
        for j in range(1, N):
            if (X[i,j], Y[i,j]) in np.column_stack((x, gamma3)):
                # Compute the normal projection onto the boundary curve gamma3
                xp = X[i,j]
                yp = Y[i,j]
                roots = np.roots([1, yp-2*xp, xp-2*yp])
                r = np.real(roots[np.abs(roots - xp).argmin()])
                xq = r
                yq = 1 - r**2
                # Compute the value of f at the projected point
                H[i-1,j-1] = f(xq, yq)

    # Construct the matrix A for the linear system
    H2 = H.ravel()
    A = diags([-1, -1, 4, -1, -1], [-N+1, -1, 0, 1, N-1], shape=((N-1)**2, (N-1)**2), format='csr')/h**2

    # Apply the boundary conditions
    b = np.zeros(((N-1)**2,))
    for i in range(1, N):
        for j in range(1, N):
            if i == 1: # boundary curve gamma1
                b[(i-1)*(N-1)+j-1] = gamma1[j-1]
            elif i == N: # boundary curve gamma2
                b[(i-2)*(N-1)+j-1] = gamma2[j-1]
            elif j == 1: # boundary curve y=0
                b[(i-2)*(N-1)+j-1] = (1/h**2)*gamma3[i]
            elif j == N: # boundary curve y=1
                b[(i-2)*(N-1)+j-1] = (1/h**2)*gamma3[i]
            else: # interior points
                b[(i-2)*(N-1)+j-1] = H2[(i-2)*(N-1)+j-1]

    # Solve the linear system using sparse matrix solver
    T = spsolve(A, b)

    # Reshape the solution vector into a 2D array and add the boundary conditions
    T = np.insert(T.reshape((N-1, N-1)), 0, gamma1, axis=0)
    T = np.insert(T, 0, gamma2, axis=0)
    T = np.insert(T, N-1, gamma3, axis=1)
    
    return X, Y, T

X, Y, T = poisson(f, N)

# Plot the solution as a surface plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X[:,-1], Y[:,-1], T.T, cmap='jet', linewidth=0, antialiased=False)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('T')
plt.title('Solution to ∇ · (κ∇T) = ∆T with fatten boundary approach')
plt.show()