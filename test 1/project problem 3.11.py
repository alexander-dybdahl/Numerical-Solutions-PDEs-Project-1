import numpy as np
import matplotlib.pyplot as plt

# Define the domain and grid parameters
x_min, x_max = 0, 1
y_min, y_max = 0, 1
N = 100 # number of grid points in each direction
dx = (x_max - x_min) / (N - 1)
dy = (y_max - y_min) / (N - 1)

# Define the boundary curves
gamma_1 = np.array([(x, 0) for x in np.linspace(x_min, x_max, N)])
gamma_2 = np.array([(0, y) for y in np.linspace(y_min, y_max, N)])
gamma_3 = np.array([(x, 1 - x**2) for x in np.linspace(x_min, x_max, N)])

# Define the fatten boundary distance
d = dx/2

# Define the coordinates of the fattened boundary points
x_fattened = []
y_fattened = []

# Loop over all grid points to find the normal extensions to the fattened points
for i in range(N):
    for j in range(N):
        if (gamma_3[i,0] - x_min)**2 + (gamma_3[i,1] - y_min)**2 <= d**2:
            # Compute the normal vector at the current boundary point
            nx = (gamma_3[i+1,1] - gamma_3[i-1,1])/(2*dx)
            ny = (gamma_3[i-1,0] - gamma_3[i+1,0])/(2*dy)
            norm = np.sqrt(nx**2 + ny**2)
            nx /= norm
            ny /= norm
            
            # Compute the coordinates of the fattened point using the cubic equation
            xp, yp = gamma_3[i]
            r = np.roots([2*d**3, 1-2*yp*d, xp])[0]
            xf = xp + r*nx
            yf = yp + r*ny
            x_fattened.append(xf)
            y_fattened.append(yf)

# Plot the domain and the normal extensions to the fattened points
plt.plot(gamma_1[:,0], gamma_1[:,1], 'k')
plt.plot(gamma_2[:,0], gamma_2[:,1], 'k')
plt.plot(gamma_3[:,0], gamma_3[:,1], 'k')
plt.plot(x_fattened, y_fattened, 'r.')
plt.xlabel('x')
plt.ylabel('y')
plt.axis('equal')
plt.show()