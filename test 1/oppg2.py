import numpy as np
import matplotlib.pyplot as plt

# Define domain and grid
nx = ny = 50
x = np.linspace(0, 1, nx)
y = np.linspace(0, 1, ny)
X, Y = np.meshgrid(x, y)
h = x[1] - x[0]

# Define boundary curves
gamma1 = np.array([np.zeros(nx), np.zeros(nx)])
gamma2 = np.array([np.zeros(ny), np.zeros(ny)])
gamma3 = np.array([x, 1 - x**2])

# Define Dirichlet boundary conditions
Tleft = np.zeros(ny)
Tright = np.zeros(ny)
Tbottom = np.zeros(nx)
Ttop = np.zeros(nx) + 1

# Fatten boundary using normal extension
def normal_extension(x, y, gamma):
    n = np.array([y - np.roll(y, 1), np.roll(x, 1) - x])
    n = np.divide(n, np.sqrt(n[0]**2 + n[1]**2))
    x_ext = x + h * n[0]
    y_ext = y + h * n[1]
    gamma_ext = np.vstack([np.flip(x_ext), np.flip(y_ext)])
    return np.hstack([gamma_ext, gamma])

gamma1_ext = normal_extension(x, np.zeros(nx), gamma1)
gamma2_ext = normal_extension(np.zeros(ny), y, gamma2)
gamma3_ext = normal_extension(x, 1 - x**2, gamma3)

# Combine boundary curves
gamma = np.hstack([gamma1_ext, gamma2_ext, gamma3_ext])

# Define function f
def f(x, y):
    return np.sin(x)

# Define the exact solution
def u(x, y, a=1, r=2):
    return -np.sin(x)

# Set up linear system
A = np.zeros((nx * ny, nx * ny))
b = np.zeros(nx * ny)

for i in range(nx):
    for j in range(ny):
        k = i + j * nx
        A[k, k] = -4
        x, y = j*h, i*h
        if i > 0:
            A[k, k-1] = 1
            b[k] -= Tleft[j]
        if i < nx-1:
            A[k, k+1] = 1
            b[k] -= Tright[j]
        if j > 0:
            A[k, k-nx] = 1
            b[k] -= Tbottom[i]
        if j < ny-1:
            A[k, k+nx] = 1
            b[k] -= Ttop[i]
            
# Solve linear system
T = np.linalg.solve(A, b).reshape((ny, nx))

# Plot solution
plt.figure(figsize=(8, 8))
plt.contourf(X, Y, T, cmap='coolwarm')
plt.colorbar()
plt.plot(gamma1[0], gamma1[1], 'k')
plt.plot(gamma2[0], gamma2[1], 'k')
plt.plot(gamma3[0], gamma3[1], 'k')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Solution of Poisson Equation with Dirichlet boundary conditions')
plt.show()