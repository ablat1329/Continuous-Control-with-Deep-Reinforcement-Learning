import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import gymnasium as gym
import panda_gym
import torch
import imageio
from sb3_contrib import HER
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.evaluation import evaluate_policy

# ===========================================
# Configuration
# ===========================================
class Config:
    ENV_NAME = "PandaPickAndPlace-v3"
    REWARD_TYPE = "sparse"  # HER works best with sparse rewards
    TOTAL_TIMESTEPS = 5_000_000  # adjust as needed
    N_EVAL_EPISODES = 5
    EVAL_FREQ = 50_000
    CHECKPOINT_FREQ = 200_000
    SAVE_DIR = "renders"
    MODEL_DIR = "models"
    LOG_DIR = "./tensorboard_logs/"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    ENT_COEF = 0.2  # fixed entropy coefficient
    POLICY_NET = [400, 300, 300]  # large network for robot arm

os.makedirs(Config.SAVE_DIR, exist_ok=True)
os.makedirs(Config.MODEL_DIR, exist_ok=True)
os.makedirs(Config.LOG_DIR, exist_ok=True)

# ===========================================
# Evaluation + GIF Callback
# ===========================================
class EvalCallback(BaseCallback):
    def __init__(self, eval_env, eval_freq=Config.EVAL_FREQ, n_eval_episodes=Config.N_EVAL_EPISODES,
                 save_dir=Config.SAVE_DIR, verbose=1):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.save_dir = save_dir
        self.best_mean_reward = -float('inf')

    def _on_step(self):
        if self.n_calls % self.eval_freq == 0:
            mean_reward, std_reward = evaluate_policy(self.model, self.eval_env, n_eval_episodes=self.n_eval_episodes, render=False)

            # Success rate
            success_count = 0
            for _ in range(self.n_eval_episodes):
                obs, _ = self.eval_env.reset()
                done = False
                while not done:
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = self.eval_env.step(action)
                    done = terminated or truncated
                if info.get("is_success", False):
                    success_count += 1
            success_rate = success_count / self.n_eval_episodes

            # Log
            print(f"\n[EVAL] Step {self.num_timesteps:,}: mean_reward={mean_reward:.2f}, success_rate={success_rate*100:.1f}%")

            # Save best model
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                self.model.save(os.path.join(Config.MODEL_DIR, "best_model"))
                print(f"[INFO] New best model saved!")

            # Create GIF
            self._create_gif()

        return True

    def _create_gif(self):
        frames = []
        obs, _ = self.eval_env.reset()
        for i in range(400):
            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = self.eval_env.step(action)
            if i % 10 == 0:
                frame = self.eval_env.render()
                frames.append(frame)
            if terminated or truncated:
                break
        gif_path = os.path.join(self.save_dir, f"eval_step_{self.num_timesteps}.gif")
        imageio.mimsave(gif_path, frames, fps=10)
        print(f"[INFO] Saved evaluation GIF: {gif_path}")

# ===========================================
# SAC + HER Training
# ===========================================
def train_sac_her():
    print("Starting SAC + HER training...")
    # Single environment for HER
    train_env = gym.make(Config.ENV_NAME, reward_type=Config.REWARD_TYPE, render_mode="rgb_array")
    eval_env = gym.make(Config.ENV_NAME, reward_type=Config.REWARD_TYPE, render_mode="rgb_array")

    # Wrap SAC with HER
    model = HER(
        "MultiInputPolicy",
        env=train_env,
        model_class=SAC,
        model_kwargs=dict(
            learning_rate=1e-3,
            buffer_size=1_000_000,
            batch_size=256,
            tau=0.05,
            gamma=0.95,
            train_freq=4,
            gradient_steps=4,
            ent_coef=Config.ENT_COEF,
            policy_kwargs=dict(net_arch=Config.POLICY_NET),
            tensorboard_log=Config.LOG_DIR,
            device=Config.DEVICE,
            verbose=1,
        ),
        n_sampled_goal=8,
        goal_selection_strategy="future",
        online_sampling=True,
    )

    # Callbacks
    eval_callback = EvalCallback(eval_env)
    checkpoint_callback = CheckpointCallback(save_freq=Config.CHECKPOINT_FREQ, save_path=Config.MODEL_DIR, name_prefix="sac_her_checkpoint")

    model.learn(total_timesteps=Config.TOTAL_TIMESTEPS, callback=[eval_callback, checkpoint_callback], progress_bar=True)
    model.save(os.path.join(Config.MODEL_DIR, "sac_her_final"))
    return model, train_env, eval_env

# ===========================================
# Run
# ===========================================
if __name__ == "__main__":
    model, train_env, eval_env = train_sac_her()

    # Final evaluation
    mean_reward, success_rate = evaluate_policy(model, eval_env, n_eval_episodes=20, render=False), None
    print(f"Final Evaluation - Mean Reward: {mean_reward}, Success Rate: {success_rate}")
    train_env.close()
    eval_env.close()
