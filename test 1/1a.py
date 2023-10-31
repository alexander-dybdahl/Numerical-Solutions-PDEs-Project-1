import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.sparse import diags, bmat
from scipy.sparse.linalg import spsolve
from scipy.linalg import solve_banded

def get_bnd(M, N):
    bnd = np.ones((M, N))
    bnd[1:-1,1:-1] = np.zeros((M-2, N-2))
    return bnd

def poisson(f, g, a=1, r=2, M = 100, x_end=1, y_end=2):
    R = a / (1+r**2)
    d = 1 / (1+r**2)

    x = np.linspace(0, x_end, M+1, endpoint=True)
    y = np.linspace(0, y_end, M+1, endpoint=True)

    X, Y = np.meshgrid(x, y)


    bnd = get_bnd(M+1, M+1)

    h = x[1] - x[0]
    k = y[1] - y[0]

    U = g(X, Y)*bnd 

    F = f(X, Y)[1:-1,1:-1]
    F = F.ravel()

    Q = (
        np.roll(np.roll(U, 1, 0), 1, 1) +
        np.roll(np.roll(U, -1, 0), -1, 1) +
        a*np.roll(U, 1, 1) +
        a*np.roll(U, -1, 1)   
    ) 
    q = (Q[1:-1, 1:-1]*get_bnd(M-1, M-1)).ravel() / h**2

    C = 2*(a+1)*np.ones((M-1)**2)

    V = -a*np.ones((M-1)**2 - 1)
    V[M-2::M-1] = 0

    H = -np.ones((M-1)**2 - M)
    H[M-2::M-1] = 0

    A = diags([H, V, C, V, H], [-M, -1, 0, 1, M], ((M-1)**2, (M-1)**2), "csc") / h**2


    U[1:-1, 1:-1] = spsolve(A,q + F).reshape(M-1, M-1)


    return X, Y, U

def g1(x, y):
    return x*y*(x-1)*(y-2)

def g2(x, y):
    return x**2 - y**2

def f1(x, y, a=1, r=2):
    return -(2*(a+1)*y**2 + 2*r**2*x**2 + 8*r*x*y -4*(a + 1 + r)*y - (8*r + 2*r**2)*x + 4*r)

def f2(x, y):
    return 4*np.ones_like(x)

def g3(x, y):
    return np.sin(np.pi*x)*np.sin(np.pi* y)

def f3(x, y, a=1, r=2):
    return -(2*np.pi**2*r*np.cos(np.pi*x)*np.cos(np.pi*y) - (1 + a)*np.pi**2*np.sin(np.pi*x)*np.sin(np.pi*y) - (np.pi*r)**2 * np.sin(np.pi*x)*np.sin(np.pi* y))


M = 100

X, Y, U = poisson(f2, g2, M=M)

ax = plt.subplot(projection="3d")
ax.plot_surface(Y, X, U, cmap=cm.coolwarm)
ax.view_init(azim=30)              # Rotate the figure
plt.xlabel('y')
plt.ylabel('x')
plt.title(f"Scheme, M = {M}")

plt.show()

# ax = plt.subplot(projection="3d")
# ax.plot_surface(Y, X, g1(X, Y), cmap=cm.coolwarm)
# ax.view_init(azim=30)              # Rotate the figure
# plt.xlabel('y')
# plt.ylabel('x')
# plt.title("Exact solution")

# plt.show()

ax = plt.subplot(projection="3d")
ax.plot_surface(Y, X, U-g2(X, Y), cmap=cm.coolwarm)
ax.view_init(azim=30)              # Rotate the figure
plt.xlabel('y')
plt.ylabel('x')
plt.title("Error")

plt.show()


# convergence rate - x direction

M = 20
num = 5
Econv = np.zeros(num)
Hconv = np.zeros(num)
ax = plt.subplot()

for i in range(num):

    X, Y, U = poisson(f1, g1, M=M)
    
    x = np.linspace(0, 1, M+1, endpoint=True)
    y = np.linspace(0, 1, M+1, endpoint=True)

    Econv[i] = np.max(np.abs(U - g1(X,Y)))
    Hconv[i] = x[1] - x[0]
    
    M *= 2
    
order = np.polyfit(np.log(Hconv),np.log(Econv),1)[0]

ax.loglog(Hconv, Econv, ".", label=f"p = {order:.2f}")
ax.set_xlabel("h")
ax.set_ylabel("error")
ax.legend()

plt.show()

