import numpy as np
import matplotlib.pyplot as plt

# Define the function for the curve
def f(x):
    return 1 - x**2

def normal_extension(xp, yp):
    # Find the closest point on the curve to the given point
    x_closest = xp
    y_closest = 1 - xp ** 2

    # Find the slope of the tangent line at the closest point
    m_tangent = -2 * xp

    # Find the slope of the normal line
    m_normal = -1 / m_tangent

    # Find the y-intercept of the normal line
    b = y_closest - m_normal * x_closest

    # Find the x-coordinate of the point at a distance of 1 from the curve
    r = np.sqrt((xp - x_closest) ** 2 + (yp - y_closest) ** 2) + 1
    xq = (r ** 2 / (1 + m_normal ** 2)) ** 0.5
    xq += x_closest if xp > x_closest else -xq + 2 * x_closest

    # Find the y-coordinate of the point on the normal line at xq
    yq = m_normal * xq + b

    return xq, yq

# Generate some x values for plotting
x_vals = np.linspace(0, 1, 100)

# Plot the curve
plt.plot(x_vals, f(x_vals), label='Curve')

# Plot some normal extension points
for xp in [0.2, 0.4, 0.6, 0.8]:
    yp = f(xp)
    xq, yq = normal_extension(xp, yp)
    plt.plot([xp, xq], [yp, yq], 'r--', label='Normal extension')

plt.xlim([0, 1])
plt.ylim([0, 1])
plt.legend()
plt.show()