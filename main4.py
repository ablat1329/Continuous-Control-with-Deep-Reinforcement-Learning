import gym
import numpy as np
from ddpg.agent import Agent
import imageio
import pathlib
import os
import matplotlib.pyplot as plt

# Create environment
env = gym.make('LunarLanderContinuous-v2')
np.random.seed(0)

# Initialize DDPG agent
agent = Agent(
    lr_actor=0.000025,
    lr_critic=0.00025,
    input_dims=[8],
    tau=0.001,
    batch_size=64,
    layer1_size=400,
    layer2_size=300,
    n_actions=2
)

score_history = []

# Folder to save images/GIFs
img_path = 'images'
pathlib.Path(img_path).mkdir(parents=True, exist_ok=True)

# Training loop
for i in range(1000):
    obs = env.reset()
    done = False
    score = 0
    frame_set = []
    record = (i % 5 == 0)  # Record every 100 episodes

    while not done:
        act = agent.choose_action(obs)
        next_state, reward, done, info = env.step(act)

        # Store experience and learn
        agent.memory.push(obs, act, reward, next_state, int(done))
        agent.learn()
        score += reward

        # Capture frame for GIF
        if record:
            frame = env.render(mode='rgb_array')  # headless rendering
            frame_set.append(frame)

        obs = next_state

    # Save GIF
    if record and frame_set:
        gif_path = os.path.join(img_path, f'eps-{i}.gif')
        imageio.mimsave(gif_path, frame_set, fps=30)

    score_history.append(score)

    # Logging
    print("==============================")
    print(f"Episode: {i}")
    print(f"Score: {score}")
    print(f"Last 100 avg: {np.mean(score_history[-100:]):.2f}")

    # Save models and plot every 50 episodes
    if i % 50 == 0:
        agent.save_models()
        plt.figure()
        plt.plot(score_history)
        plt.xlabel('Episodes')
        plt.ylabel('Score')
        plt.grid()
        plt.savefig(os.path.join(img_path, "score_fig.png"))
        plt.close()

# Close environment
env.close()
print("Training completed!")
