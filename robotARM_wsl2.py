import os
import pathlib
import gymnasium as gym
import panda_gym
import imageio
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

print("Did u start?")
# --- Config ---
env_id = "PandaPickAndPlace-v3"

# --- Create environment ---
# Force PyBullet to use DIRECT if GUI fails
try:
    env = gym.make(env_id, render_mode="human")
    print("✅ Environment created with GUI (human render mode)")
except Exception as e:
    print("⚠️ GUI mode failed, falling back to DIRECT mode")
    os.environ["PYBULLET_EGL"] = "1"  # use EGL headless rendering if available
    env = gym.make(env_id, render_mode="rgb_array")
    print("✅ Environment created in rgb_array mode (headless safe)")

obs, info = env.reset()
print("Environment reset successful.")

# --- Config ---
total_timesteps = 50000
record_interval = 50  # record every N episodes
video_folder = "videos"
os.makedirs(video_folder, exist_ok=True)

# --- Initialize PPO agent ---
model = PPO("MultiInputPolicy", env, verbose=1)

# --- Training loop with video recording ---
score_history = []
episode = 0
while model.num_timesteps < total_timesteps:
    obs = env.reset()[0]
    done = False
    score = 0
    frames = []

    while not done:
        action, _ = model.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        score += reward

        # Always record frames for video
        frames.append(env.render())

        # Train on the step
        model.learn(total_timesteps=1, reset_num_timesteps=False)

    score_history.append(score)
    episode += 1

    # Save video every N episodes
    if episode % record_interval == 0:
        out_path = os.path.join(video_folder, f"episode_{episode}.gif")
        imageio.mimsave(out_path, frames, fps=30)
        print(f"Saved video: {out_path}")

    # Plot scores
    plt.figure(figsize=(8, 4))
    plt.plot(score_history)
    plt.xlabel("Episodes")
    plt.ylabel("Score")
    plt.grid(True)
    plt.savefig(os.path.join(video_folder, "score_plot.png"))
    plt.close()

    print(f"Episode {episode}, Score: {score}, Avg(last 50): {sum(score_history[-50:])/min(50, len(score_history))}")

# --- Save final model ---
model.save("ppo_panda_arm")
env.close()
print("Training complete. Model saved.")
