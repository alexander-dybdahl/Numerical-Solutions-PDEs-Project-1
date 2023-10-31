import numpy as np
import matplotlib.pyplot as plt

# Set up the domain and boundary
x_min, x_max = 0, 1
y_min, y_max = 0, 1
num_points = 50
X, Y = np.meshgrid(np.linspace(x_min, x_max, num_points), np.linspace(y_min, y_max, num_points))
boundary = ((X == x_min) | (Y == y_min) | (Y == 1 - X**2)).astype(int)

# Set up the Dirichlet boundary condition
T_bc = np.zeros((num_points, num_points))
T_bc[:, -1] = 1  # T(x,1) = 1

# Set up the initial guess and tolerance
T_guess = np.zeros((num_points, num_points))
T_tolerance = 1e-4

# Set up the diffusion coefficient
kappa = np.ones_like(T_guess)

# Fatten the boundary
boundary_fat = np.zeros_like(boundary)
for i in range(num_points):
    for j in range(num_points):
        if boundary[i, j]:
            # Find the projection of (i, j) onto gamma_3
            def r_func(r):
                return i + r * (1 - 2 * j / num_points) - 2 * r**3
            r = np.roots([2, 0, -2 * j / num_points, 1 - i])[np.isreal(np.roots([2, 0, -2 * j / num_points, 1 - i]))][0]
            xq, yq = r, 1 - r**2
            # Determine the indices of the closest point to the projection
            i_q, j_q = int(round(xq * (num_points - 1))), int(round(yq * (num_points - 1)))
            # Set the corresponding point in the fat boundary to 1
            boundary_fat[i_q, j_q] = 1

# Iterate until the solution converges
while True:
    T_new = T_guess.copy()
    for i in range(1, num_points - 1):
        for j in range(1, num_points - 1):
            if not boundary_fat[i, j]:
                T_new[i, j] = 0.25 * (T_guess[i + 1, j] + T_guess[i - 1, j] + T_guess[i, j + 1] + T_guess[i, j - 1])
    T_new *= (1 - boundary_fat)
    T_new += T_bc * boundary_fat
    if np.max(np.abs(T_new - T_guess)) < T_tolerance:
        break
    T_guess = T_new

# Calculate the exact solution and the error
T_exact = X * (1 - Y**2)
T_error = T_new - T_exact

# Plot the results
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
axs[0].set_title("Numerical Solution")
axs[0].imshow(T_new, cmap="jet", origin="lower", extent=(x_min, x_max, y_min, y_max))
axs[0].contour(X, Y, boundary, levels=[0.5], colors="black", linewidths=2)
axs[0].set_xlabel("x")
axs[0].set_ylabel("y")
axs[1].set_title("Exact Solution")
axs[1].imshow(T_exact, cmap="jet", origin="lower", extent=(x_min, x_max, y_min, y_max))

plt.show()