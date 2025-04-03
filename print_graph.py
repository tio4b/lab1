import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
import locale

locale.setlocale(locale.LC_NUMERIC, 'de_DE.UTF-8')

def function(p):
    x, y = p
    return x** 2 + y ** 2

def load_data(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    data = []
    for line in lines:
        parts = line.strip().split(' , ')
        x = locale.atof(parts[0])
        y = locale.atof(parts[1])
        data.append([x, y])
    return np.array(data)

data = load_data("gradient_log.csv")
x_coords, y_coords = data[:, 0], data[:, 1]
z_coords = function([x_coords, y_coords])

x = np.linspace(min(x_coords) - 0.5, max(x_coords) + 0.5, 25)
y = np.linspace(min(y_coords) - 0.5, max(y_coords) + 0.5, 25)
X, Y = np.meshgrid(x, y)
Z = function([X, Y])

fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')

ls = LightSource(azdeg=135, altdeg=45)
rgb = ls.shade(Z, cmap=plt.cm.viridis, vert_exag=0.1, blend_mode='soft')
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, facecolors=rgb, alpha=0.5)

ax.plot(x_coords, y_coords, z_coords, '-', color='#FF4500', linewidth=2, label='Trajectory')
ax.scatter(x_coords, y_coords, z_coords, s=30, c='#00FFFF', edgecolor='black', linewidth=1, alpha=1, label='Temp point')
ax.scatter(x_coords[0], y_coords[0], z_coords[0], s=200, c='green', edgecolor='black', linewidth=2, alpha=1, label='Start point')
ax.scatter(x_coords[-1], y_coords[-1], z_coords[-1], s=200, c='red', edgecolor='black', linewidth=2, alpha=1, label='End point')

offset_z = np.min(Z) - 10
ax.contour(X, Y, Z, zdir='z', offset=offset_z, levels=10, cmap='plasma', alpha=0.3)

ax.set_xlim(min(x_coords) - 0.5, max(x_coords) + 0.5)
ax.set_ylim(min(y_coords) - 0.5, max(y_coords) + 0.5)
ax.set_zlim(offset_z, max(z_coords) + 5)
ax.view_init(elev=35, azim=-50)

ax.set_xlabel('X', fontsize=12)
ax.set_ylabel('Y', fontsize=12)
ax.set_zlabel('Z', fontsize=12)
ax.set_title('Trajectory of gradient descent', fontsize=14)
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()
