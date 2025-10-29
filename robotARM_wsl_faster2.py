import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"      # disables oneDNN optimizations
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"       # hides all TF info/warning logs
import imageio
import gymnasium as gym
import panda_gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy

# --- CONFIGURE DISPLAY FOR WSL (optional) ---
if "DISPLAY" not in os.environ or not os.environ["DISPLAY"]:
    try:
        with open("/etc/resolv.conf") as f:
            line = next((l for l in f if "nameserver" in l), None)
            if line:
                ip = line.strip().split()[1]
                os.environ["DISPLAY"] = f"{ip}:0"
                print(f"[INFO] DISPLAY set to {os.environ['DISPLAY']}")
    except Exception as e:
        print(f"[WARN] Could not set DISPLAY automatically: {e}")

# --- EVALUATION + GIF CALLBACK ---
class EvalGifCallback(BaseCallback):
    def __init__(self, eval_env, eval_freq=50000, n_eval_episodes=3, save_dir="renders", verbose=1):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def _on_step(self):
        if self.n_calls % self.eval_freq == 0:
            mean_reward, std_reward = evaluate_policy(
                self.model, self.eval_env, n_eval_episodes=self.n_eval_episodes, render=False
            )
            if self.verbose > 0:
                print(f"\n[INFO] Step {self.num_timesteps}: mean_reward={mean_reward:.2f} ± {std_reward:.2f}")

            # Record GIF
            frames = []
            obs, _ = self.eval_env.reset()
            for i in range(300):  # short rollout
                action, _ = self.model.predict(obs)
                obs, reward, done, truncated, info = self.eval_env.step(action)
                frame = self.eval_env.render()
                if i % 10 == 0:  # fewer frames = smaller gif
                    frames.append(frame)
                if done or truncated:
                    obs, _ = self.eval_env.reset()

            gif_path = os.path.join(self.save_dir, f"eval_step_{self.num_timesteps}.gif")
            imageio.mimsave(gif_path, frames, fps=10)
            print(f"[INFO] Saved evaluation GIF: {gif_path}\n")

        return True

# --- CREATE TRAINING ENVIRONMENT ---
env = gym.make("PandaPickAndPlace-v3", render_mode="rgb_array")
eval_env = gym.make("PandaPickAndPlace-v3", render_mode="rgb_array")

# --- DEFINE MODEL WITH TENSORBOARD LOGGING ---
model = PPO(
    "MultiInputPolicy",
    env,
    verbose=1,
    tensorboard_log="./logs/",
)

# --- TRAIN WITH CALLBACK ---
callback = EvalGifCallback(eval_env, eval_freq=50000, save_dir="renders")
model.learn(total_timesteps=10_000_000, callback=callback)

# --- FINAL EVALUATION ---
mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=5)
print(f"\n✅ Final evaluation: mean_reward={mean_reward:.2f} ± {std_reward:.2f}")

env.close()
eval_env.close()

#  tensorboard --logdir ./logs/
