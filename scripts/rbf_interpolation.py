import numpy as np
from scipy.interpolate import Rbf
import matplotlib.pyplot as plt

# Define the radius and number of points
radius = 5
num_points = 100

# Generate points equally distributed within a circle
angles = np.linspace(0, 2 * np.pi, num_points)
radii = np.linspace(0, radius, num_points)
x = radii * np.cos(angles)
y = radii * np.sin(angles)
z = np.sin(np.sqrt(x**2 + y**2))  # Example function value at each point

print(f"Shapes are x:{x.shape}, y:{y.shape},z:{z.shape}")

# Define the target grid where we want to interpolate
xi = np.linspace(-radius, radius, 200)
yi = np.linspace(-radius, radius, 200)
XI, YI = np.meshgrid(xi, yi)
RI = np.sqrt(XI**2 + YI**2)

# Filter out points outside the circle (optional)
mask = RI <= radius

# Flatten and filter points within the circle for interpolation
x_flat = x.flatten()
y_flat = y.flatten()
z_flat = z.flatten()
print(f"Shapes are x_flat:{x_flat.shape}, y_flat:{y_flat.shape},z_flat:{z_flat.shape}")

# Perform RBF interpolation
rbf = Rbf(x_flat, y_flat, z_flat, function="multiquadric")
ZI = rbf(XI, YI)

# Apply the mask to keep values only within the circle
ZI[~mask] = np.nan

# Plot the results
plt.figure(figsize=(8, 6))
plt.contourf(XI, YI, ZI, cmap="viridis")
plt.title("RBF Interpolation within a Circle")
plt.colorbar()
plt.scatter(x, y, c=z, edgecolor="k", label="Sample Points")
plt.legend()
plt.show()
