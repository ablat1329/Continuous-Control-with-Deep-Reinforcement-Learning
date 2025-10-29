"""
Enhanced Deep RL for Panda Robot Arm Pick and Place
Combines PPO multi-core training with SAC+HER option
Includes TensorBoard logging, GIF rendering, and evaluation
"""

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"      # disables oneDNN optimizations
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"       # hides all TF info/warning logs
import multiprocessing
import imageio
import gymnasium as gym
import panda_gym
from stable_baselines3 import PPO, SAC, HerReplayBuffer
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize, SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
import torch


# ==============================================
#  CONFIGURATION
# ==============================================
class Config:
    # Choose algorithm: "PPO" or "SAC_HER"
    ALGORITHM = "PPO"  # Change to "SAC_HER" to use SAC with HER

    # Training
    TOTAL_TIMESTEPS = 10_000_000
    NUM_ENVS = max(2, multiprocessing.cpu_count() - 1)

    # Environment
    ENV_NAME = "PandaPickAndPlace-v3"
    REWARD_TYPE = "dense"  # "dense" or "sparse"

    # Callbacks
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
#  DISPLAY CONFIG (for WSL users)
# ==============================================
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


# ==============================================
#  ENHANCED CALLBACK: Eval + GIF + Success Rate
# ==============================================
class EnhancedEvalCallback(BaseCallback):
    def __init__(self, eval_env, eval_freq=100000, n_eval_episodes=5,
                 save_dir="renders", verbose=1):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.best_mean_reward = -float('inf')

    def _on_step(self):
        if self.n_calls % self.eval_freq == 0:
            # Evaluate policy
            mean_reward, std_reward = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                render=False
            )

            # Calculate success rate
            success_count = 0
            for _ in range(self.n_eval_episodes):
                obs, _ = self.eval_env.reset()
                done = False
                while not done:
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = self.eval_env.step(action)
                    done = terminated or truncated
                if info.get('is_success', False):
                    success_count += 1

            success_rate = success_count / self.n_eval_episodes

            # Log to TensorBoard
            self.logger.record("eval/mean_reward", mean_reward)
            self.logger.record("eval/std_reward", std_reward)
            self.logger.record("eval/success_rate", success_rate)

            print(f"\n{'='*60}")
            print(f"[EVAL] Step {self.num_timesteps:,}")
            print(f"  Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}")
            print(f"  Success Rate: {success_rate*100:.1f}%")
            print(f"{'='*60}\n")

            # Save best model
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                self.model.save(os.path.join(Config.MODEL_DIR, "best_model"))
                print(f"[INFO] New best model saved! (reward: {mean_reward:.2f})")

            # Create evaluation GIF
            self._create_eval_gif()

        return True

    def _create_eval_gif(self):
        """Generate a GIF of the current policy"""
        frames = []
        obs, _ = self.eval_env.reset()

        for i in range(400):
            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = self.eval_env.step(action)

            if i % 10 == 0:  # Capture every 10th frame
                frame = self.eval_env.render()
                frames.append(frame)

            if terminated or truncated:
                break

        gif_path = os.path.join(self.save_dir, f"eval_step_{self.num_timesteps}.gif")
        imageio.mimsave(gif_path, frames, fps=10)
        print(f"[INFO] Saved evaluation GIF: {gif_path}")


# ==============================================
#  PPO TRAINING (Multi-core)
# ==============================================
def train_ppo():
    print(f"\n{'='*60}")
    print(f"🤖 Training with PPO (Multi-core)")
    print(f"{'='*60}")
    print(f"Environments: {Config.NUM_ENVS}")
    print(f"Device: {Config.DEVICE}")
    print(f"Total Steps: {Config.TOTAL_TIMESTEPS:,}\n")

    env_kwargs = {
        "reward_type": Config.REWARD_TYPE,
        "render_mode": "rgb_array"
    }

    # Create parallel training environments
    train_env = make_vec_env(
        Config.ENV_NAME,
        n_envs=Config.NUM_ENVS,
        env_kwargs=env_kwargs,
        vec_env_cls=SubprocVecEnv,
    )

    # Normalize observations & rewards
    train_env = VecNormalize(
        train_env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0
    )

    # Evaluation environment
    eval_env = gym.make(Config.ENV_NAME, **env_kwargs)

    # PPO hyperparameters (optimized for robotic manipulation)
    policy_kwargs = dict(
        log_std_init=-2,
        ortho_init=False,
        net_arch=dict(pi=[256, 256], vf=[256, 256])  # Larger networks
    )

    model = PPO(
        "MultiInputPolicy",
        train_env,
        verbose=1,
        n_steps=2048,
        batch_size=512,
        n_epochs=20,
        gamma=0.98,
        gae_lambda=0.95,
        ent_coef=0.001,
        vf_coef=0.5,
        learning_rate=3e-4,
        clip_range=0.2,
        max_grad_norm=0.5,
        policy_kwargs=policy_kwargs,
        tensorboard_log=Config.LOG_DIR,
        device=Config.DEVICE
    )

    # Setup callbacks
    eval_callback = EnhancedEvalCallback(
        eval_env,
        eval_freq=Config.EVAL_FREQ,
        n_eval_episodes=Config.N_EVAL_EPISODES,
        save_dir=Config.SAVE_DIR
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=Config.CHECKPOINT_FREQ // Config.NUM_ENVS,  # Adjust for parallel envs
        save_path=Config.MODEL_DIR,
        name_prefix="ppo_checkpoint"
    )

    # Train
    print("🚀 Starting PPO training...\n")
    model.learn(
        total_timesteps=Config.TOTAL_TIMESTEPS,
        callback=[eval_callback, checkpoint_callback],
        progress_bar=True
    )

    # Save final model
    train_env.save(os.path.join(Config.MODEL_DIR, "vec_normalize.pkl"))
    model.save(os.path.join(Config.MODEL_DIR, "ppo_final"))

    return model, train_env, eval_env


# ==============================================
#  SAC + HER TRAINING (Single environment)
# ==============================================
def train_sac_her():
    print(f"\n{'='*60}")
    print(f"🤖 Training with SAC + HER")
    print(f"{'='*60}")
    print(f"Device: {Config.DEVICE}")
    print(f"Total Steps: {Config.TOTAL_TIMESTEPS:,}\n")

    env_kwargs = {
        "reward_type": Config.REWARD_TYPE,
        "render_mode": "rgb_array"
    }

    # Create training environment
    train_env = gym.make(Config.ENV_NAME, **env_kwargs)

    # Evaluation environment
    eval_env = gym.make(Config.ENV_NAME, **env_kwargs)

    # SAC with HER (Hindsight Experience Replay)
    model = SAC(
        "MultiInputPolicy",
        train_env,
        replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs=dict(
            n_sampled_goal=4,
            goal_selection_strategy="future",
        ),
        learning_rate=1e-3,
        buffer_size=1_000_000,
        batch_size=256,
        gamma=0.95,
        tau=0.05,
        train_freq=1,
        gradient_steps=1,
        policy_kwargs=dict(net_arch=[256, 256, 256]),
        verbose=1,
        tensorboard_log=Config.LOG_DIR,
        device=Config.DEVICE,
    )

    # Setup callbacks
    eval_callback = EnhancedEvalCallback(
        eval_env,
        eval_freq=Config.EVAL_FREQ,
        n_eval_episodes=Config.N_EVAL_EPISODES,
        save_dir=Config.SAVE_DIR
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=Config.CHECKPOINT_FREQ,
        save_path=Config.MODEL_DIR,
        name_prefix="sac_her_checkpoint"
    )

    # Train
    print("🚀 Starting SAC+HER training...\n")
    model.learn(
        total_timesteps=Config.TOTAL_TIMESTEPS,
        callback=[eval_callback, checkpoint_callback],
        log_interval=10,
        progress_bar=True
    )

    # Save final model
    model.save(os.path.join(Config.MODEL_DIR, "sac_her_final"))

    return model, train_env, eval_env


# ==============================================
#  FINAL EVALUATION
# ==============================================
def final_evaluation(model, eval_env, is_vec_env=False):
    print(f"\n{'='*60}")
    print("🏁 FINAL EVALUATION")
    print(f"{'='*60}\n")

    if is_vec_env:
        # Load normalization stats for PPO
        eval_env = VecNormalize.load(
            os.path.join(Config.MODEL_DIR, "vec_normalize.pkl"),
            eval_env
        )
        eval_env.training = False
        eval_env.norm_reward = False

    mean_reward, std_reward = evaluate_policy(
        model,
        eval_env,
        n_eval_episodes=20,
        render=False
    )

    # Calculate success rate
    success_count = 0
    for ep in range(20):
        obs, _ = eval_env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            done = terminated or truncated
        if info.get('is_success', False):
            success_count += 1

    success_rate = success_count / 20

    print(f"Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}")
    print(f"Success Rate: {success_rate*100:.1f}%")
    print(f"\n{'='*60}\n")

    return mean_reward, success_rate


# ==============================================
#  MAIN ENTRY POINT
# ==============================================
if __name__ == "__main__":
    # Create directories
    os.makedirs(Config.SAVE_DIR, exist_ok=True)
    os.makedirs(Config.MODEL_DIR, exist_ok=True)
    os.makedirs(Config.LOG_DIR, exist_ok=True)

    print(f"\n{'='*60}")
    print("🦾 PANDA ROBOT ARM - DEEP RL TRAINING")
    print(f"{'='*60}")
    print(f"Algorithm: {Config.ALGORITHM}")
    print(f"Environment: {Config.ENV_NAME}")
    print(f"Reward Type: {Config.REWARD_TYPE}")
    print(f"Total Timesteps: {Config.TOTAL_TIMESTEPS:,}")
    print(f"Device: {Config.DEVICE}")

    # Train based on selected algorithm
    if Config.ALGORITHM == "PPO":
        model, train_env, eval_env = train_ppo()
        is_vec_env = True
    elif Config.ALGORITHM == "SAC_HER":
        model, train_env, eval_env = train_sac_her()
        is_vec_env = False
    else:
        raise ValueError(f"Unknown algorithm: {Config.ALGORITHM}")

    print("\n✅ Training complete!")

    # Final evaluation
    final_evaluation(model, eval_env, is_vec_env)

    print("💾 All models and stats saved.")
    print(f"\n📊 View TensorBoard logs with:")
    print(f"   tensorboard --logdir={Config.LOG_DIR}")

    # Cleanup
    train_env.close()
    eval_env.close()
