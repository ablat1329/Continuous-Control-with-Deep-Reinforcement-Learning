
# %%
import gymnasium as gym
import panda_gym
from stable_baselines3 import PPO
import time
import math

#%%
# Create the environment with rendering
env = gym.make("PandaPickAndPlace-v3", render_mode="human")

# Initialize PPO agent
model = PPO("MultiInputPolicy", env, verbose=1)

# Train the agent
model.learn(total_timesteps=50000)

# Reset environment
obs, _ = env.reset()

# Function to update camera view (orbit around the arm)
def update_camera(env, step, radius=1.5, height=1.0, speed=0.02):
    # Compute angle for circular motion
    angle = speed * step
    camera_x = radius * math.cos(angle)
    camera_y = radius * math.sin(angle)
    camera_z = height
    # Set camera position and look at the robot's base
    env.render(camera_position=[camera_x, camera_y, camera_z],
               camera_target=[0, 0, 0])

# Run the trained agent with dynamic camera
for step in range(500):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)

    # Update camera
    update_camera(env, step)

    # Slow down for smooth visualization
    time.sleep(0.02)

    if terminated or truncated:
        obs, _ = env.reset()

env.close()
