"""
Final Enhanced Deep RL for Panda Robot Arm (Pick & Place)
✅ Supports PPO (multi-core) and SAC + HER (single env)
✅ Works with Stable-Baselines3 v2.7.0 + Gymnasium
✅ Includes TensorBoard logging, evaluation GIFs, and success tracking
"""

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"   # Disable oneDNN noise
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"    # Suppress TensorFlow logs

import multiprocessing
import gymnasium as gym
import panda_gym
import imageio
import torch

from stable_baselines3 import PPO, SAC, HerReplayBuffer
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecNormalize,
)
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.evaluation import evaluate_policy


# ==============================================
# CONFIGURATION
# ==============================================
class Config:
    # Algorithm selection
    ALGORITHM = "SAC_HER"  # or "PPO"

    # Training parameters
    TOTAL_TIMESTEPS = 5_000_000
    NUM_ENVS = max(2, multiprocessing.cpu_count() - 1)

    # SAC-specific
    USE_FIXED_ENTROPY = False   # Let entropy auto-tune (better stability)
    FIXED_ENTROPY_COEF = 0.2

    # Environment
    ENV_NAME = "PandaPickAndPlace-v3"
    REWARD_TYPE = "sparse"  # Sparse reward is best for HER

    # Logging and evaluation
    EVAL_FREQ = 100_000
    CHECKPOINT_FREQ = 500_000
    N_EVAL_EPISODES = 5

    # Directories
    SAVE_DIR = "renders"
    MODEL_DIR = "models"
    LOG_DIR = "./tensorboard_logs/"

    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ==============================================
# ENHANCED CALLBACK: EVAL + GIF + SUCCESS RATE
# ==============================================
class EnhancedEvalCallback(BaseCallback):
    def __init__(self, eval_env, eval_freq, n_eval_episodes, save_dir, verbose=1):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.best_success = 0.0

    def _on_step(self):
        if self.n_calls % self.eval_freq == 0:
            mean_reward, std_reward = evaluate_policy(
                self.model, self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                render=False
            )

            # Compute success rate
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
            self.logger.record("eval/mean_reward", mean_reward)
            self.logger.record("eval/success_rate", success_rate)

            print(f"\n{'='*60}")
            print(f"[EVAL] Step {self.num_timesteps:,}")
            print(f"Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}")
            print(f"Success Rate: {success_rate*100:.1f}%")
            print(f"{'='*60}\n")

            if success_rate > self.best_success:
                self.best_success = success_rate
                self.model.save(os.path.join(Config.MODEL_DIR, "best_model"))
                print(f"[INFO] ✅ New best model saved! Success: {success_rate*100:.1f}%")

            self._create_eval_gif()
        return True

    def _create_eval_gif(self):
        frames = []
        obs, _ = self.eval_env.reset()
        for i in range(400):
            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = self.eval_env.step(action)
            if i % 10 == 0:
                frame = self.eval_env.render()
                frames.append(frame)
            if terminated or truncated:
                break
        gif_path = os.path.join(self.save_dir, f"eval_step_{self.num_timesteps}.gif")
        imageio.mimsave(gif_path, frames, fps=10)
        print(f"[INFO] 🎥 Saved GIF: {gif_path}")


# ==============================================
# PPO TRAINING (MULTI-CORE)
# ==============================================
def train_ppo():
    print(f"\n{'='*60}\n🤖 Training PPO (multi-core)\n{'='*60}")
    env_kwargs = {"reward_type": Config.REWARD_TYPE, "render_mode": "rgb_array"}

    train_env = make_vec_env(
        Config.ENV_NAME,
        n_envs=Config.NUM_ENVS,
        env_kwargs=env_kwargs,
        vec_env_cls=SubprocVecEnv,
    )

    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True)

    eval_env = gym.make(Config.ENV_NAME, **env_kwargs)

    model = PPO(
        "MultiInputPolicy",
        train_env,
        n_steps=2048,
        batch_size=512,
        n_epochs=20,
        gamma=0.98,
        gae_lambda=0.95,
        ent_coef=0.001,
        learning_rate=3e-4,
        policy_kwargs=dict(
            log_std_init=-2,
            ortho_init=False,
            net_arch=dict(pi=[256, 256], vf=[256, 256]),
        ),
        verbose=1,
        tensorboard_log=Config.LOG_DIR,
        device=Config.DEVICE,
    )

    eval_callback = EnhancedEvalCallback(
        eval_env, Config.EVAL_FREQ, Config.N_EVAL_EPISODES, Config.SAVE_DIR
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=Config.CHECKPOINT_FREQ // Config.NUM_ENVS,
        save_path=Config.MODEL_DIR,
        name_prefix="ppo_checkpoint"
    )

    model.learn(
        total_timesteps=Config.TOTAL_TIMESTEPS,
        callback=[eval_callback, checkpoint_callback],
        progress_bar=True,
    )

    train_env.save(os.path.join(Config.MODEL_DIR, "vec_normalize.pkl"))
    model.save(os.path.join(Config.MODEL_DIR, "ppo_final"))
    return model, train_env, eval_env


# ==============================================
# SAC + HER TRAINING (SINGLE ENV)
# ==============================================
def train_sac_her():
    print(f"\n{'='*60}\n🤖 Training SAC + HER\n{'='*60}")
    env_kwargs = {"reward_type": Config.REWARD_TYPE, "render_mode": "rgb_array"}

    train_env = gym.make(Config.ENV_NAME, **env_kwargs)
    eval_env = gym.make(Config.ENV_NAME, **env_kwargs)

    if Config.USE_FIXED_ENTROPY:
        ent_coef = Config.FIXED_ENTROPY_COEF
        print(f"[INFO] Using fixed entropy coef: {ent_coef}")
    else:
        ent_coef = "auto"
        print("[INFO] Using automatic entropy tuning")

    model = SAC(
        "MultiInputPolicy",
        train_env,
        replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs=dict(
            n_sampled_goal=8,
            goal_selection_strategy="future",
            # online_sampling=True,
        ),
        learning_rate=1e-3,
        buffer_size=1_000_000,
        learning_starts=2000,
        batch_size=256,
        gamma=0.96,
        tau=0.05,
        ent_coef=ent_coef,
        train_freq=4,
        gradient_steps=4,
        policy_kwargs=dict(net_arch=[256, 256, 256]),
        verbose=1,
        tensorboard_log=Config.LOG_DIR,
        device=Config.DEVICE,
    )

    eval_callback = EnhancedEvalCallback(
        eval_env, Config.EVAL_FREQ, Config.N_EVAL_EPISODES, Config.SAVE_DIR
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=Config.CHECKPOINT_FREQ,
        save_path=Config.MODEL_DIR,
        name_prefix="sac_her_checkpoint"
    )

    model.learn(
        total_timesteps=Config.TOTAL_TIMESTEPS,
        callback=[eval_callback, checkpoint_callback],
        log_interval=10,
        progress_bar=True,
    )

    model.save(os.path.join(Config.MODEL_DIR, "sac_her_final"))
    return model, train_env, eval_env


# ==============================================
# FINAL EVALUATION
# ==============================================
def final_evaluation(model, eval_env, is_vec_env=False):
    print(f"\n{'='*60}\n🏁 FINAL EVALUATION\n{'='*60}")
    if is_vec_env:
        eval_env = VecNormalize.load(
            os.path.join(Config.MODEL_DIR, "vec_normalize.pkl"), eval_env
        )
        eval_env.training = False
        eval_env.norm_reward = False

    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10)
    success = 0
    for _ in range(10):
        obs, _ = eval_env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            done = terminated or truncated
        if info.get("is_success", False):
            success += 1
    print(f"Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}")
    print(f"Success Rate: {success * 10:.1f}%")
    print("=" * 60)


# ==============================================
# MAIN
# ==============================================
if __name__ == "__main__":
    os.makedirs(Config.SAVE_DIR, exist_ok=True)
    os.makedirs(Config.MODEL_DIR, exist_ok=True)
    os.makedirs(Config.LOG_DIR, exist_ok=True)

    print(f"\n{'='*60}\n🦾 PANDA ROBOT ARM TRAINING\n{'='*60}")
    print(f"Algorithm: {Config.ALGORITHM}")
    print(f"Device: {Config.DEVICE}")

    if Config.ALGORITHM == "PPO":
        model, train_env, eval_env = train_ppo()
        is_vec = True
    else:
        model, train_env, eval_env = train_sac_her()
        is_vec = False

    final_evaluation(model, eval_env, is_vec)

    print("\n✅ Training complete. View logs with:")
    print(f"tensorboard --logdir={Config.LOG_DIR}")


# This code somehow works, and in tensorboard you can see the results, SAC_9
#
