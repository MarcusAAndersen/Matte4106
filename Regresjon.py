import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

data = np.array([
    [100, 25, 25, 25, 25, 25, 25],
    [100, 60, 43, 37, 31, 25, 25],
    [100, 65, 51, 42, 40, 30, 25],
    [100, 65, 40, 40, 30, 29, 25],
    [100, 65, 43, 39, 35, 32, 25],
    [100, 65, 45, 41, 36, 32, 25],
    [100, 65, 47, 43, 37, 33, 25],
    [100, 70, 45, 42, 37, 32, 25],
    [100, 72, 45, 43, 38, 31, 25],
    [100, 76, 52, 45, 39, 30, 25],
    [100, 76, 54, 45, 40, 30, 25]
])

tid = np.array([0, 0.5, 1, 1.5, 4, 5, 6, 7, 10, 15, 20])
pos = np.array([0, 15, 20, 25, 30, 50, 75])

# Create an array to store all points
points = []

# Iterate through data and create points
for i, t in enumerate(tid):
    for j, p in enumerate(pos):
        temp = data[i, j]  # Access temperature value from data array
        points.append([t, p, temp])

points = np.array(points)

# Perform Gaussian Process Regression
X = points[:, :2]  # Features (time and position)
y = points[:, 2]   # Target (temperature)

kernel = 1.0 * RBF(length_scale=1.0, length_scale_bounds=(1e-1, 10.0)) + WhiteKernel(noise_level=1e-5)
model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
model.fit(X, y)

# Plotting
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Generate grid of points for plotting surface
t_grid, p_grid = np.meshgrid(np.linspace(tid.min(), tid.max(), 100), np.linspace(pos.min(), pos.max(), 100))
X_pred = np.column_stack((t_grid.ravel(), p_grid.ravel()))
y_pred, sigma = model.predict(X_pred, return_std=True)
y_pred = y_pred.reshape(t_grid.shape)

# Clip predicted temperatures to ensure they don't fall below 25 and exceed 100
y_pred_clipped = np.clip(y_pred, 25, 100)

# Plot surface
ax.plot_surface(t_grid, p_grid, y_pred_clipped, cmap='viridis', alpha=0.)

# Scatter plot original data points
ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=points[:, 2], cmap='viridis')

# Set labels and title
ax.set_xlabel('Time (min)')
ax.set_ylabel('Position')
ax.set_zlabel('Temperature')
ax.set_title('Temperature Distribution over Time and Position (Gaussian Process Regression)')

plt.show()