import gym
import numpy as np
from ddpg.agent import Agent
import matplotlib.pyplot as plt
import pathlib

# ============================
# Environment & Agent Setup
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
# Training Loop (No rendering)
# ============================
for i in range(1000):
    done = False
    score = 0
    obs = env.reset()

    while not done:
        act = agent.choose_action(obs)
        next_state, reward, done, info = env.step(act)
        agent.memory.push(obs, act, reward, next_state, int(done))
        agent.learn()
        score += reward
        obs = next_state

    score_history.append(score)
    last_100_avg = np.mean(score_history[-100:])

    print("==============================")
    print(f"Episode: {i}")
    print(f"Score: {score:.2f}")
    print(f"Last 100 avg: {last_100_avg:.2f}")

    # Save model + score plot every 50 episodes
    if i % 50 == 0:
        agent.save_models()
        plt.plot(score_history)
        plt.xlabel('Episodes')
        plt.ylabel('Score')
        plt.grid(True)
        plt.savefig(pathlib.Path(img_path) / "score_fig.png")
        plt.close()

# ============================
# Cleanup
# ============================
env.close()
print("[INFO] Training complete. Fully headless, no rendering called.")
