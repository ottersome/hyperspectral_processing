import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt

# Sample data: A 2D grid of values

x = np.linspace(0, 10, 5)  # 5 points between 0 and 10
y = np.linspace(0, 10, 5)
z = np.sin(x[:, None] * y[None, :])  # Create a sample surface

# Create a mesh grid

X, Y = np.meshgrid(x, y)

# Points where we want to interpolate

x_new = np.linspace(0, 10, 50)
y_new = np.linspace(0, 10, 50)
X_new, Y_new = np.meshgrid(x_new, y_new)

# Bilinear interpolation

f_bilinear = interpolate.interp2d(x, y, z, kind="linear")
Z_bilinear = f_bilinear(x_new, y_new)

# Bicubic interpolation

f_bicubic = interpolate.interp2d(x, y, z, kind="cubic")
Z_bicubic = f_bicubic(x_new, y_new)

# Plotting the results

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.contourf(X_new, Y_new, Z_bilinear, cmap="viridis")
plt.title("Bilinear Interpolation")
plt.colorbar()

plt.subplot(1, 2, 2)
plt.contourf(X_new, Y_new, Z_bicubic, cmap="viridis")
plt.title("Bicubic Interpolation")
plt.colorbar()

plt.show()
