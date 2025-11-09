import matplotlib.pyplot as plt
import numpy as np

# Number of triangles
n = 20

# Initialize
x, y = [0], [0]
angle = 0

# Generate spiral points
for i in range(1, n + 1):
    angle += np.arctan(1 / np.sqrt(i))
    x.append(np.sqrt(i) * np.cos(angle))
    y.append(np.sqrt(i) * np.sin(angle))

# Plot setup
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_aspect('equal')

# Draw spiral lines
for i in range(1, len(x)):
    ax.plot([x[i-1], x[i]], [y[i-1], y[i]], color='blue', linestyle='dashed', linewidth=0.8)
    ax.plot([0, x[i]], [0, y[i]], color='blue', linestyle='dotted', linewidth=0.8)
    ax.text(x[i]*1.05, y[i]*1.05, f'√{i}', fontsize=10, color='blue')

# Draw red outer line
ax.plot(x, y, color='red', linewidth=2)

# Draw concentric dotted circles
radii = np.arange(1, np.ceil(np.sqrt(n)) + 2)
for r in radii:
    circle = plt.Circle((0, 0), r, color='green', fill=False, linestyle='dotted', alpha=0.5)
    ax.add_artist(circle)
    ax.text(r * np.cos(np.pi/12), r * np.sin(np.pi/12), str(int(r)), color='green', fontsize=8)

# Adjustments
ax.set_xlim(-6, 6)
ax.set_ylim(-6, 6)
ax.axis('off')
ax.set_title("Art of Mathematics — The Pythagorean Spiral", fontsize=14)

plt.show()
