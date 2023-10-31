import numpy as np
import matplotlib.pyplot as plt

# Define the function for the curve
def f(x):
    return 1 - x**2

def normal_extension(x, y):
    roots = np.roots([-2, 0, 1-2*y, x])
    print(roots)
    r = np.real(roots[np.argmin(np.imag(roots)**2)])
    return r, 1-r**2

N = 20
x = np.linspace(0, 1, N+1)
y = np.linspace(0, 1, N+1)
h = 1/N

x_vals = np.linspace(0, 1, 100)

plt.plot(x_vals, f(x_vals), label='Curve')

for i in range(N):
    for j in range(N):
        if y[j] < f(x[i]):
            plt.plot(x[i], y[j], 'bx')
        else:
            plt.plot(x[i], y[j], 'rx')

xp = 19*h
yp = 7*h
xq, yq = normal_extension(xp, yp)
print(xq, yq)
plt.plot([xp, xq], [yp, yq], 'r--', label='Normal extension')

plt.xlim([0, 1])
plt.ylim([0, 1])
plt.legend()
plt.show()