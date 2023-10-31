import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

def normal_extension(x, y):
    roots = np.roots([-2, 0, 1-2*y, x])
    r = np.real(roots[np.argmin(np.imag(roots)**2)])
    return r, 1-r**2

def poisson(f, g, M = 100):
    
    x = np.linspace(0, 1, M+1, endpoint=True)
    y = np.linspace(0, 1, M+1, endpoint=True)
    X, Y = np.meshgrid(x, y)

    h = x[1] - x[0]
    d = h/2
    
    gamma = lambda x: 1 - x**2

    A = np.zeros(((M+1)**2, (M+1)**2))
    b = np.zeros((M+1)**2)

    for i in range(M+1):
        for j in range(M+1):
            k = i*(M+1) + j
            x = i*h
            y = j*h
            if i == 0 or j == 0: # dirichlet on the boundary
                A[k,k] = 1
                b[k] = g(x, y)
            elif y >= 1 - x**2 - d: # fattened boundary
                A[k, k] = 1
                xq, yq = normal_extension(x, y) # ghost point
                b[k] = g(xq, yq)
            else: # points inside the domain
                A[k,k] = -4/h**2
                A[k,k-1] = 1/h**2
                A[k,k+1] = 1/h**2
                A[k,k-(M+1)] = 1/h**2
                A[k,k+(M+1)] = 1/h**2
                b[k] = f(x, y)

    T = np.linalg.solve(A, b).reshape((M+1, M+1)).T

    interior = np.zeros_like(X)
    interior[Y <= 1 - X**2] = 1
    T[interior == 0] = X[interior == 0] = Y[interior == 0] = None
    
    return X, Y, T

M = 100
f = lambda x, y: -np.sin(x)
g = lambda x, y: np.sin(x)

X, Y, T = poisson(f, g, M = 100)

fig = plt.figure(figsize=plt.figaspect(0.33))

ax1 = fig.add_subplot(1, 3, 1, projection="3d")
ax1.plot_surface(X, Y, T, cmap=cm.coolwarm)
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('T')
ax1.set_title('Solution to ∆T = -f')

ax2 = fig.add_subplot(1, 3, 2, projection='3d')
surf = ax2.plot_surface(X, Y, g(X, Y), cmap=cm.coolwarm)
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_zlabel('T')
ax2.set_title('Exact solution to ∆T = -f')

ax3 = fig.add_subplot(1, 3, 3, projection='3d')
surf = ax3.plot_surface(X, Y, T - g(X, Y), cmap=cm.coolwarm)
ax3.set_xlabel('x')
ax3.set_ylabel('y')
ax3.set_zlabel('T')
ax3.set_title('Error')

plt.show()


fig, ax1 = plt.subplots(1, 1, figsize=(12, 5))

# Error in x
num = 4
Econv = np.zeros(num)
Hconv = np.zeros(num)
M = 8
for i in range(num):
    X, Y, T = poisson(f, g, M = M)
    print(M)
    Econv[i] = np.max(np.nan_to_num(np.abs(T - g(X, Y))))
    Hconv[i] = X[0, 1] - X[0, 0]
    M *= 2

order = np.polyfit(np.log(Hconv),np.log(Econv),1)[0]

ax1.loglog(Hconv, Econv, ".", label=f"p = {order:.2f}")
ax1.set_xlabel("h")
ax1.set_ylabel("error")
ax1.legend()

plt.show()
