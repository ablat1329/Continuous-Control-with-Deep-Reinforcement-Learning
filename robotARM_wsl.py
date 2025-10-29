import os
# import gym
import gymnasium as gym
import panda_gym
from stable_baselines3 import PPO

# --- CONFIGURE DISPLAY FOR WSL ---
if "DISPLAY" not in os.environ or not os.environ["DISPLAY"]:
    # Try to auto-detect Windows host IP and export display
    try:
        with open("/etc/resolv.conf") as f:
            line = next((l for l in f if "nameserver" in l), None)
            if line:
                ip = line.strip().split()[1]
                os.environ["DISPLAY"] = f"{ip}:0"
                print(f"[INFO] DISPLAY set to {os.environ['DISPLAY']}")
    except Exception as e:
        print(f"[WARN] Could not set DISPLAY automatically: {e}")

# --- CREATE ENVIRONMENT ---
env = gym.make("PandaPickAndPlace-v3", render_mode="human")

# --- DEFINE & TRAIN MODEL ---
model = PPO("MultiInputPolicy", env, verbose=1)
model.learn(total_timesteps=10000)

# --- EVALUATE MODEL ---
obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs)
    obs, reward, done, info = env.step(action)

    env.render()

    if done:
        obs = env.reset()

env.close()
