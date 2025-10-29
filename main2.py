import os
import subprocess
import time
import gym
import numpy as np
import pathlib
import imageio
import matplotlib.pyplot as plt
import cv2
from ddpg.agent import Agent

# ============================
# Display / OpenGL detection
# ============================
def detect_display():
    display = os.environ.get("DISPLAY")
    if display:
        try:
            subprocess.check_output(["xdpyinfo"], stderr=subprocess.STDOUT)
            import pyglet
            try:
                config = pyglet.gl.Config(double_buffer=True)
                window = pyglet.window.Window(width=1, height=1, visible=False, config=config)
                window.close()
                print(f"[INFO] OpenGL context available → windowed mode.")
                os.environ["LIBGL_ALWAYS_INDIRECT"] = "1"
                return "window"
            except Exception as e:
                print(f"[WARN] Cannot create GL context: {e}")
        except Exception:
            print("[INFO] DISPLAY exists but X server not reachable.")
    print("[INFO] Using EGL offscreen rendering.")
    os.environ["PYOPENGL_PLATFORM"] = "egl"
    os.environ.pop("DISPLAY", None)
    return "egl"

render_mode = detect_display()

# ============================
# Environment and agent
# ============================
env = gym.make('LunarLanderContinuous-v2')
np.random.seed(0)

agent = Agent(
    lr_actor=0.000025, lr_critic=0.00025, input_dims=[8],
    tau=0.001, batch_size=64, layer1_size=400, layer2_size=300, n_actions=2
)

score_history = []
img_path = 'images'
pathlib.Path(img_path).mkdir(parents=True, exist_ok=True)

# ============================
# Helper: Overlay info on frame
# ============================
def overlay_info(frame, score, fps, last_100_avg):
    # Determine color based on performance
    if score >= last_100_avg:
        color = (0, 255, 0)  # green = improving
    elif score >= last_100_avg * 0.8:
        color = (0, 255, 255)  # yellow = average
    else:
        color = (0, 0, 255)  # red = poor

    text_score = f"Score: {score:.2f}"
    text_fps = f"FPS: {fps:.1f}"
    cv2.putText(frame, text_score, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                1, color, 2, cv2.LINE_AA)
    cv2.putText(frame, text_fps, (10, 70), cv2.FONT_HERSHEY_SIMPLEX,
                1, (255, 255, 255), 2, cv2.LINE_AA)
    return frame

# ============================
# Training loop
# ============================
for i in range(1000):
    done = False
    score = 0
    obs = env.reset()
    frame_set = []
    record = (i % 100 == 0)
    prev_time = time.time()

    last_100_avg = np.mean(score_history[-100:]) if len(score_history) > 0 else 0

    while not done:
        act = agent.choose_action(obs)
        next_state, reward, done, info = env.step(act)
        agent.memory.push(obs, act, reward, next_state, int(done))
        agent.learn()
        score += reward

        # Render frame for GIF
        try:
            frame = env.render(mode='rgb_array')
            if record:
                frame_set.append(frame)
            # Live preview using OpenCV
            if render_mode == "egl":
                current_time = time.time()
                fps = 1.0 / (current_time - prev_time)
                prev_time = current_time
                frame_preview = overlay_info(frame.copy(), score, fps, last_100_avg)
                frame_preview = cv2.resize(frame_preview, (640, 480))
                cv2.imshow("LunarLander (live preview)", cv2.cvtColor(frame_preview, cv2.COLOR_RGB2BGR))
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    done = True
        except Exception as e:
            print(f"[WARN] Render failed (offscreen ok): {e}")

        obs = next_state

    # Save GIF
    if record and frame_set:
        gif_path = os.path.join(img_path, f'eps-{i}.gif')
        imageio.mimsave(gif_path, frame_set, fps=30)
        print(f"[INFO] Saved GIF: {gif_path}")

    score_history.append(score)
    last_100_avg = np.mean(score_history[-100:])

    print("==============================")
    print(f"Episode: {i}")
    print(f"Score: {score:.2f}")
    print(f"Last 100 avg: {last_100_avg:.2f}")

    # Save model and plot every 50 episodes
    if i % 50 == 0:
        agent.save_models()
        plt.plot(score_history)
        plt.xlabel('episodes')
        plt.ylabel('score')
        plt.grid(True)
        plt.savefig(os.path.join(img_path, "score_fig.png"))
        plt.close()

# Cleanup
env.close()
if render_mode == "egl":
    cv2.destroyAllWindows()
print("[INFO] Training complete.")
