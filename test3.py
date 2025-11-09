import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider

# --- Parameters ---
n = 17                # Max number of triangles
circle_step = 0.25
dot_density = 200

# --- Precompute spiral points ---
angles = [0]
x, y = [0], [0]
for i in range(1, n + 1):
    angle = sum(np.arctan(1 / np.sqrt(k)) for k in range(1, i + 1))
    angles.append(angle)
    x.append(np.sqrt(i) * np.cos(angle))
    y.append(np.sqrt(i) * np.sin(angle))

# --- Figure setup ---
fig, ax = plt.subplots(figsize=(8, 8))
plt.subplots_adjust(bottom=0.15)
ax.set_aspect("equal")
ax.axis("off")

# --- Dotted circular grid ---
max_r = np.sqrt(n) + 1
radii = np.arange(circle_step, max_r, circle_step)
for r in radii:
    theta = np.linspace(0, 2 * np.pi, dot_density)
    ax.plot(r * np.cos(theta), r * np.sin(theta),
            linestyle='None', marker='.', color='black', alpha=0.25, markersize=1)

# --- Static green outer circle ---
theta = np.linspace(0, 2 * np.pi, 1000)
ax.plot(max_r * np.cos(theta), max_r * np.sin(theta), color='green', lw=2)

# --- Dynamic elements (will update during animation) ---
blue_lines = []
red_lines = []
labels = []

# --- Function to update animation ---
def update(frame):
    # Clear previous dynamic objects
    for l in blue_lines + red_lines + labels:
        l.remove()
    blue_lines.clear()
    red_lines.clear()
    labels.clear()

    # Draw lines up to current frame
    for i in range(1, frame + 1):
        b_line, = ax.plot([0, x[i]], [0, y[i]], color='blue', lw=0.8)
        r_line, = ax.plot([x[i - 1], x[i]], [y[i - 1], y[i]], color='red', lw=2)
        t = ax.text(x[i] * 1.05, y[i] * 1.05, f"√{i}", color='blue', fontsize=10)
        blue_lines.append(b_line)
        red_lines.append(r_line)
        labels.append(t)

    return blue_lines + red_lines + labels

# --- Animation ---
ani = FuncAnimation(fig, update, frames=n + 1, interval=700, blit=False, repeat=True)

# --- Slider to control step manually ---
ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])
slider = Slider(ax_slider, 'Step', 1, n, valinit=1, valstep=1)

def slider_update(val):
    frame = int(slider.val)
    update(frame)
    fig.canvas.draw_idle()

slider.on_changed(slider_update)

# --- Center point label ---
ax.scatter(0, 0, color='blue', s=40)
ax.text(0.1, -0.1, "A", color='blue', fontsize=10)

ax.set_xlim(-max_r - 1, max_r + 1)
ax.set_ylim(-max_r - 1, max_r + 1)
ax.set_title("Art of Mathematics — The Pythagorean Spiral", fontsize=14, pad=15)

plt.show()
