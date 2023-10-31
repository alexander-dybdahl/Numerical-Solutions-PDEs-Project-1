import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.sparse import diags, bmat
from scipy.sparse.linalg import spsolve
from scipy.linalg import solve_banded

def gamma(x):
    return 1 - x**2

def gamma_inv(y):
    return np.sqrt(1 - y)

def poisson(f, g, M = 100):
    x = np.linspace(0, 1, M+1, endpoint=True)
    y = np.linspace(0, 1, M+1, endpoint=True)

    h = x[1] - x[0]

    X, Y = np.meshgrid(x, y)

    interior = np.zeros_like(X)

    boundary_hit = np.zeros_like(X)
    boundary_hit[X == 0] = 1
    boundary_hit[Y == 0] = 1
    boundary_hit[Y == gamma(X)] = 1
    boundary_hit[X == gamma_inv(Y)] = 1

    interior = np.zeros_like(X)
    interior[Y < gamma(X)] = 1
    interior[boundary_hit == 1] = 0

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

    # plt.pcolormesh(X, Y, interior_no_bnd + north_boundary*2 + boundary_hit*3)
    # plt.pcolormesh(X, Y,interior_no_bnd + north_boundary + east_boundary + boundary_hit*5)
    # plt.colorbar()
    # plt.show()

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

    U[exterior == 1] = None

    X[exterior == 1] = None
    Y[exterior == 1] = None

    return X, Y, U

def g(x, y):
    return x**2 - y**2

def f(x, y):
    return np.zeros_like(x)

def g1(x, y):
    return np.sin(x)

def f1(x, y):
    return -np.sin(x)

M = 100

X, Y, U = poisson(f1, g1, M = M)

ax = plt.subplot(projection="3d")
ax.plot_surface(X, Y, U, cmap=cm.coolwarm)
ax.view_init(azim=30)              # Rotate the figure
plt.xlabel('x')
plt.ylabel('y')
plt.title(f"Scheme, M = {M}")

plt.show()

ax = plt.subplot(projection="3d")
ax.plot_surface(X, Y, g1(X, Y), cmap=cm.coolwarm)
ax.view_init(azim=30)              # Rotate the figure
plt.xlabel('x')
plt.ylabel('y')
plt.title("Exact")

plt.show()

ax = plt.subplot(projection="3d")
ax.plot_surface(X, Y, U - g1(X, Y), cmap=cm.coolwarm)
ax.view_init(azim=30)              # Rotate the figure
plt.xlabel('x')
plt.ylabel('y')
plt.title("Error")

plt.show()


fig, ax1 = plt.subplots(1, 1, figsize=(12, 5))

# Error in x
num_x = 6
Econv = np.zeros(num_x)
Hconv = np.zeros(num_x)
M = 8
for i in range(num_x):
    X, Y, U = poisson(f1, g1, M = M)
    print(M)

    Econv[i] = np.max(np.nan_to_num(np.abs(U - g1(X, Y))))
    Hconv[i] = X[0, 1] - X[0, 0]

    M *= 2

order = np.polyfit(np.log(Hconv),np.log(Econv),1)[0]

ax1.loglog(Hconv, Econv, ".", label=f"p = {order:.2f}")
ax1.set_xlabel("h")
ax1.set_ylabel("error")
ax1.legend()

plt.show()