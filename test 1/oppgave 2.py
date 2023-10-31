import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.sparse import diags
from mpl_toolkits.mplot3d import Axes3D

x_max = 1
y_max = 1
M = 100

def gamma(x):
    return 1 - x**2

def normal_extension(xp, yp, f):
    roots = np.roots([1, yp-2*xp, xp-2*yp])
    r = np.real(roots[np.abs(roots - xp).argmin()])
    xq = r
    yq = 1 - r**2
    return f(xq, yq)

def poisson(f, g, M = 100):

    x = np.linspace(0, 1, M+1, endpoint=True)
    y = np.linspace(0, 1, M+1, endpoint=True)

    h = x[1] - x[0]
    
    X, Y = np.meshgrid(x, y)
    
    interior = np.zeros_like(X)

    # mark where the grid points are at the boundary curve
    boundary_hit = np.zeros_like(X)
    boundary_hit[X == 0] = 1
    boundary_hit[Y == 0] = 1
    boundary_hit[Y == gamma(X)] = 1
    boundary_hit[X == gamma_inv(Y)] = 1

    # mark the interior points
    interior = np.zeros_like(X)
    interior[Y < gamma(X)] = 1
    interior[boundary_hit == 1] = 0

    # mark the exterior points
    exterior = np.zeros_like(X)
    exterior[Y > gamma(X)] = 1
    exterior[boundary_hit == 1] = 0
    
    
    dist_y = (np.roll(gamma(X), 1, 0) - Y) / h
    dist_y[interior + np.roll(exterior, -1, 0) != 2] = 0

    dist_x = (np.roll(gamma_inv(Y), 1, 1) - X) / h
    dist_x[interior + np.roll(exterior, -1, 1) != 2] = 0

    north_boundary = np.zeros_like(X)
    north_boundary[dist_y != 0] = 1
    
    east_boundary = np.zeros_like(X)
    east_boundary[dist_x != 0] = 1

    interior_no_bnd = interior.copy()
    interior_no_bnd[north_boundary == 1] = 0
    interior_no_bnd[east_boundary == 1] = 0

    A = np.zeros((M**2, M**2))
    
    F = (f(X, Y) * interior).ravel()
    
    Q = (
        g(X, Y) * boundary_hit +
        g(X, gamma(X)) * np.divide(2, dist_y*(1+dist_y), out=np.zeros_like(X), where=dist_y != 0) / h**2 +
        g(gamma_inv(Y), Y) * np.divide(2, dist_x*(1+dist_x), out=np.zeros_like(X), where=dist_x != 0) / h**2
    ).ravel()

    F = (f(X, Y) * interior).ravel()


    P = boundary_hit + exterior + (4*interior_no_bnd + north_boundary*np.divide(2, dist_y, out=np.zeros_like(X), where=dist_y != 0) + east_boundary*np.divide(2, dist_x, out=np.zeros_like(X), where=dist_x != 0)) / h**2

    N = -interior_no_bnd / h**2
    S = (-interior_no_bnd - north_boundary*np.divide(2, 1+dist_y, out=np.zeros_like(X), where=dist_y != 0)) / h**2
    E = -interior_no_bnd / h**2
    W = (-interior_no_bnd - east_boundary*np.divide(2, 1+dist_x, out=np.zeros_like(X), where=dist_x != 0) )/ h**2

    A = diags([S.ravel()[M+1:], W.ravel()[1:], P.ravel(), E.ravel()[:-1], N.ravel()[:-(M+1)]], [-(M+1), -1, 0, 1, M+1], ((M+1)**2, (M+1)**2), "csc")

    U = spsolve(A, Q - F).reshape(M+1, M+1)
    
    
    X[exterior == 1] = None
    Y[exterior == 1] = None
    
    return X, Y, T

def f(x, y):
    return -np.sin(x)

def g(x, y):
    return np.sin(x)

X, Y, T = poisson(f, g)

# Plot the solution as a surface plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, Y, T.T, cmap=cm.coolwarm, linewidth=0, antialiased=False)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('T')
plt.title('Solution to ∆T = -f with fatten boundary approach')
plt.show()

# Plot the solution as a surface plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, Y, g(X, Y), cmap=cm.coolwarm, linewidth=0, antialiased=False)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('T')
plt.title('Exact solution to ∆T = -f with fatten boundary approach')
plt.show()