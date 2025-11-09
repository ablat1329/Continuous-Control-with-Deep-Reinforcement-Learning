"""
Interactive Pythagorean (Theodorus) spiral with animation, play/pause button and slider.

Works in a normal Python environment. For Jupyter notebooks, enable an interactive backend, e.g.:
%matplotlib notebook
or
%matplotlib widget
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, Button

# --- Parameters ---
n = 17                # Max number of triangles (√1 ... √n)
circle_step = 0.25    # spacing for dotted circles
dot_density = 300     # number of dots per circle

# --- Precompute spiral points ---
angles = [0.0]
x = [0.0]
y = [0.0]
for i in range(1, n + 1):
    # angle added by triangle with legs 1 and sqrt(i)
    ang = np.arctan(1 / np.sqrt(i))
    angles.append(angles[-1] + ang)
    x.append(np.sqrt(i) * np.cos(angles[-1] + ang))
    y.append(np.sqrt(i) * np.sin(angles[-1] + ang))

# --- Plot setup ---
fig, ax = plt.subplots(figsize=(8, 8))
plt.subplots_adjust(left=0.05, right=0.95, top=0.92, bottom=0.18)
ax.set_aspect("equal")
ax.axis("off")

# --- Dotted circular grid (static) ---
max_r = np.sqrt(n) + 1.0
radii = np.arange(circle_step, max_r + 0.001, circle_step)
theta_full = np.linspace(0, 2*np.pi, dot_density)
for r in radii:
    ax.plot(r * np.cos(theta_full), r * np.sin(theta_full),
            linestyle='None', marker='.', markersize=1, alpha=0.25)

# --- Outer green circle ---
theta = np.linspace(0, 2*np.pi, 1000)
ax.plot(max_r * np.cos(theta), max_r * np.sin(theta), color='green', lw=2)

# --- Pre-create dynamic elements (so we only update them) ---
blue_radials = []
red_edges = []
sqrt_labels = []

for i in range(1, n + 1):
    # radial from origin to point i
    (bl,) = ax.plot([0, 0], [0, 0], color='blue', lw=0.9, visible=False)
    blue_radials.append(bl)
    # red edge between points i-1 and i
    (rl,) = ax.plot([0, 0], [0, 0], color='red', lw=2, visible=False)
    red_edges.append(rl)
    # label near the point
    t = ax.text(0, 0, f"√{i}", color='blue', fontsize=10, visible=False)
    sqrt_labels.append(t)

# center marker and label
ax.scatter(0, 0, color='blue', s=40)
ax.text(0.08, -0.12, "A", color='blue', fontsize=10)

ax.set_xlim(-max_r-1, max_r+1)
ax.set_ylim(-max_r-1, max_r+1)
ax.set_title("Art of Mathematics — The Pythagorean Spiral", fontsize=14, pad=12)

# --- Animation control state ---
current_frame = 1
playing = True

# --- Update function (shows up to frame `k`) ---
def show_frame(k):
    """Show triangles up to step k (1..n). If k==0, nothing visible."""
    # clamp
    k = max(0, min(n, int(k)))
    for i in range(1, n + 1):
        idx = i - 1
        if i <= k:
            # show radial
            blue_radials[idx].set_data([0, x[i]], [0, y[i]])
            blue_radials[idx].set_visible(True)
            # show red edge
            red_edges[idx].set_data([x[i-1], x[i]], [y[i-1], y[i]])
            red_edges[idx].set_visible(True)
            # show label slightly outside the point
            lx = x[i] * 1.06
            ly = y[i] * 1.06
            sqrt_labels[idx].set_position((lx, ly))
            sqrt_labels[idx].set_visible(True)
        else:
            blue_radials[idx].set_visible(False)
            red_edges[idx].set_visible(False)
            sqrt_labels[idx].set_visible(False)
    fig.canvas.draw_idle()

# --- FuncAnimation callback for smooth playback ---
def animate(frame):
    # frame will be 0..n; show that many steps
    global current_frame
    current_frame = int(frame)
    slider.set_val(current_frame)  # update slider (this will call slider_update)
    return []

# Create animation but we will control play/pause via Button
ani = FuncAnimation(fig, animate, frames=range(0, n+1), interval=700, blit=False, repeat=True)
ani.event_source.stop()  # start paused; we'll toggle

# --- Slider widget to pick number of steps manually ---
ax_slider = plt.axes([0.18, 0.06, 0.62, 0.03])
slider = Slider(ax_slider, 'Step', 0, n, valinit=current_frame, valstep=1)

def slider_update(val):
    global current_frame
    current_frame = int(val)
    show_frame(current_frame)

slider.on_changed(slider_update)

# --- Play / Pause button ---
ax_button = plt.axes([0.85, 0.02, 0.1, 0.055])
btn = Button(ax_button, 'Play/Pause')

def on_play_pause(event):
    global playing
    if playing:
        ani.event_source.stop()
        playing = False
    else:
        ani.event_source.start()
        playing = True

btn.on_clicked(on_play_pause)

# --- Initial draw ---
show_frame(current_frame)

plt.show()
