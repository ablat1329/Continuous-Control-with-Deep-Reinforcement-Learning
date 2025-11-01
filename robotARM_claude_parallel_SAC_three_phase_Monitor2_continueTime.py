#!/usr/bin/env python3
"""
SAC + HER training for PandaPickAndPlace with:
 - resume capability
 - two-phase training schedule (explore -> refine)
 - checkpointing & best-model saving
 - evaluation callback with GIF & CSV logging
 - debug_rollouts helper

Updated for stable-baselines3 2.7.0+ (HerReplayBuffer instead of HER wrapper)
"""

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import time
import glob
import json
import csv
import gymnasium as gym
import panda_gym
import imageio
import torch
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.her.her_replay_buffer import HerReplayBuffer
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.monitor import Monitor

# -------------------------
# Configuration
# -------------------------
class Config:
    ENV_NAME = "PandaPickAndPlace-v3"
    # HER works best with sparse rewards
    REWARD_TYPE = "sparse"

    # Training total (sum of phase_timesteps)
    TOTAL_TIMESTEPS = 5_000_000

    # Phase schedule: list of (timesteps, n_sampled_goal, ent_coef, lr)
    # Phase 0: broad exploration with HER (500k steps)
    # Phase 1: continued learning (1.5M steps)
    # Phase 2: fine-tuning with lower LR (3M steps)
    PHASES = [
        {"timesteps": 500_000, "n_sampled_goal": 4, "ent_coef": "auto", "learning_rate": 1e-3},
        {"timesteps": 1_500_000, "n_sampled_goal": 4, "ent_coef": "auto", "learning_rate": 8e-4},
        {"timesteps": 3_000_000, "n_sampled_goal": 4, "ent_coef": "auto", "learning_rate": 3e-4},
    ]

    # SAC model params (common)
    BUFFER_SIZE = int(1e6)
    BATCH_SIZE = 256
    TAU = 0.005  # Slower target network updates for stability
    GAMMA = 0.95  # Lower discount for sparse rewards
    TRAIN_FREQ = (1, "episode")  # Train after each episode, not each step
    GRADIENT_STEPS = -1  # Match number of steps in episode
    POLICY_NET = [256, 256, 256]  # Smaller networks often work better
    LEARNING_STARTS = 1000  # Start training after collecting some experience

    # Logging / Checkpoints / Eval
    LOG_DIR = "./tensorboard_logs/"
    MODEL_DIR = "models"
    SAVE_DIR = "renders"
    CSV_LOG = os.path.join(MODEL_DIR, "eval_history_continuous.csv")
    EVAL_FREQ = 25_000  # More frequent evaluation
    N_EVAL_EPISODES = 10  # More episodes for reliable metrics
    CHECKPOINT_FREQ = 100_000  # More frequent checkpoints

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    SEED = 42

# make folders
os.makedirs(Config.MODEL_DIR, exist_ok=True)
os.makedirs(Config.SAVE_DIR, exist_ok=True)
os.makedirs(Config.LOG_DIR, exist_ok=True)

set_random_seed(Config.SEED)

# -------------------------
# Helper to create properly wrapped environments
# -------------------------
def make_env(env_name, reward_type, render_mode="rgb_array", monitor_dir=None):
    """
    Create environment with proper Monitor wrapper for accurate logging
    """
    env = gym.make(env_name, reward_type=reward_type, render_mode=render_mode)

    # Wrap with Monitor for proper episode tracking
    if monitor_dir:
        os.makedirs(monitor_dir, exist_ok=True)
        env = Monitor(env, monitor_dir)
    else:
        # For eval envs, use Monitor without file logging
        env = Monitor(env)

    return env

# -------------------------
# Utilities: checkpoint/resume helpers
# -------------------------
def find_latest_checkpoint(prefix="sac_her_checkpoint"):
    pattern = os.path.join(Config.MODEL_DIR, f"{prefix}*")
    files = sorted(glob.glob(pattern))
    if not files:
        return None
    # find largest modified file by timestamp (approx latest)
    latest = max(files, key=os.path.getmtime)
    return latest

def save_phase_meta(phase_idx, remaining_timesteps):
    meta = {"phase_idx": phase_idx, "remaining_timesteps": remaining_timesteps, "timestamp": time.time()}
    with open(os.path.join(Config.MODEL_DIR, "train_meta_continue.json"), "w") as f:
        json.dump(meta, f)

def load_phase_meta():
    path = os.path.join(Config.MODEL_DIR, "train_meta_continue.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return None

# -------------------------
# Eval callback with GIF, CSV logging, best model saving
# -------------------------
class EvalAndSaveCallback(BaseCallback):
    def __init__(self, eval_env, eval_freq=Config.EVAL_FREQ, n_eval_episodes=Config.N_EVAL_EPISODES,
                 save_dir=Config.SAVE_DIR, verbose=1):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.save_dir = save_dir
        self.best_mean_reward = -float("inf")
        self.last_eval_step = 0  # Track last evaluation to avoid duplicates

        # init CSV if missing
        if not os.path.exists(Config.CSV_LOG):
            with open(Config.CSV_LOG, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["timesteps", "mean_reward", "std_reward", "success_rate", "mean_episode_length", "timestamp"])

    def _on_step(self) -> bool:
        # Check if we should evaluate based on total timesteps across all phases
        steps_since_last_eval = self.num_timesteps - self.last_eval_step

        if steps_since_last_eval >= self.eval_freq:
            self.last_eval_step = self.num_timesteps

            mean_reward, std_reward = evaluate_policy(self.model, self.eval_env, n_eval_episodes=self.n_eval_episodes, render=False)

            # compute deterministic success rate
            # FIXED: Track success throughout the episode, not just final step
            success_count = 0
            episode_lengths = []
            for _ in range(self.n_eval_episodes):
                obs, _ = self.eval_env.reset()
                done = False
                episode_success = False  # Track if success occurred at ANY point
                steps = 0
                while not done:
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = self.eval_env.step(action)
                    steps += 1

                    # Check success at EVERY step, not just the last one
                    if info.get("is_success", False):
                        episode_success = True

                    done = terminated or truncated

                episode_lengths.append(steps)
                if episode_success:
                    success_count += 1
            success_rate = success_count / self.n_eval_episodes
            mean_episode_length = np.mean(episode_lengths)

            # Print & log
            ts = self.num_timesteps
            print(f"\n[EVAL] Step {ts:,}: mean_reward={mean_reward:.3f} ± {std_reward:.3f}, success_rate={success_rate*100:.1f}%, ep_len={mean_episode_length:.1f}")
            self.logger.record("eval/mean_reward", mean_reward)
            self.logger.record("eval/std_reward", std_reward)
            self.logger.record("eval/success_rate", success_rate)
            self.logger.record("eval/mean_episode_length", mean_episode_length)
            self.logger.dump(ts)

            # write CSV
            with open(Config.CSV_LOG, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([ts, float(mean_reward), float(std_reward), float(success_rate), float(mean_episode_length), time.time()])

            # save best model
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                best_path = os.path.join(Config.MODEL_DIR, "best_model")
                self.model.save(best_path)
                print(f"[INFO] New best model saved: {best_path} (reward: {mean_reward:.3f})")

            # create gif of deterministic rollout
            self._create_gif()

        return True

    def _create_gif(self):
        """Create high-quality GIF with all frames and success indicator"""
        frames = []
        obs, _ = self.eval_env.reset()
        episode_info = {
            'steps': 0,
            'success': False,
            'final_distance': None,
            'grasped': False
        }

        for i in range(400):
            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = self.eval_env.step(action)

            # Capture EVERY frame for smooth video
            frames.append(self.eval_env.render())

            # Track important events
            episode_info['steps'] = i + 1
            if info.get("is_success", False):
                episode_info['success'] = True

            # Try to detect grasping (environment-dependent)
            if 'achieved_goal' in info and 'desired_goal' in info:
                achieved = np.array(info['achieved_goal'])
                desired = np.array(info['desired_goal'])
                episode_info['final_distance'] = float(np.linalg.norm(achieved - desired))

            if terminated or truncated:
                break

        if frames:
            # Add pause frames at the end (1 second = 15 frames at 15fps)
            pause_frame = frames[-1].copy()
            for _ in range(15):
                frames.append(pause_frame)

            # Determine success/failure for filename
            status = "SUCCESS" if episode_info['success'] else "FAIL"
            dist_str = f"_dist{episode_info['final_distance']:.3f}" if episode_info['final_distance'] else ""
            gif_path = os.path.join(
                self.save_dir,
                f"eval_{self.num_timesteps:07d}_{status}{dist_str}.gif"
            )

            # Save at 15 fps for better viewability (was 20 fps)
            # Duration parameter adds delay between frames for slower playback
            imageio.mimsave(gif_path, frames, fps=15, loop=0)

            print(f"[INFO] GIF saved: {gif_path}")
            print(f"       Status: {status}, Steps: {episode_info['steps']}, Distance: {episode_info['final_distance']:.4f}" if episode_info['final_distance'] else "")
        else:
            print("[WARN] No frames captured for GIF.")

# -------------------------
# High-quality visualization helper
# -------------------------
def create_detailed_visualization(model, env, n_episodes=5, save_dir="detailed_renders"):
    """
    Create high-quality videos with overlay information showing:
    - Current step
    - Distance to goal
    - Success status
    - Gripper state (if available)

    Note: Expects unwrapped environment for rendering
    """
    os.makedirs(save_dir, exist_ok=True)

    # Unwrap Monitor if present for rendering
    render_env = env.unwrapped if isinstance(env, Monitor) else env

    for ep in range(n_episodes):
        obs, info = env.reset()
        frames = []
        step = 0
        success = False

        print(f"\n--- Recording Episode {ep+1} ---")

        while step < 400:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            # Capture every single frame from unwrapped env
            frame = render_env.render()
            frames.append(frame)

            # Print progress every 10 steps
            if step % 10 == 0 and 'achieved_goal' in info and 'desired_goal' in info:
                achieved = np.array(info['achieved_goal'])
                desired = np.array(info['desired_goal'])
                dist = np.linalg.norm(achieved - desired)
                print(f"  Step {step:3d}: distance={dist:.4f}, reward={reward:.2f}")

            step += 1

            if terminated or truncated:
                success = info.get("is_success", False)
                print(f"  Episode ended: {step} steps, success={success}")
                break

        if frames:
            # Add pause at the end (2 seconds = 24 frames at 12fps)
            pause_frame = frames[-1].copy()
            for _ in range(24):
                frames.append(pause_frame)

            status = "SUCCESS" if success else "FAIL"
            gif_path = os.path.join(save_dir, f"detailed_ep{ep+1:02d}_{status}_{step}steps.gif")

            # Save at 12 fps for better viewability and smoothness balance
            imageio.mimsave(gif_path, frames, fps=12, loop=0)
            print(f"  Saved: {gif_path} ({len(frames)} frames @ 12fps, includes 2s pause)")

# -------------------------
# Debug helper: prints info & saves GIFs
# -------------------------
def debug_rollouts(model, env, n_episodes=3, save_dir="debug_renders"):
    os.makedirs(save_dir, exist_ok=True)
    successful_eps = []
    failed_eps = []

    # Unwrap Monitor if present for rendering
    render_env = env.unwrapped if isinstance(env, Monitor) else env

    for ep in range(n_episodes):
        obs, _ = env.reset()
        frames = []
        step = 0
        episode_reward = 0
        print(f"\n--- Debug Episode {ep+1} ---")

        while True:
            action, _ = model.predict(obs, deterministic=False)
            # action stats
            arr = np.array(action)
            try:
                amin, amax, amean = float(arr.min()), float(arr.max()), float(arr.mean())
            except Exception:
                amin = amax = amean = float(arr)

            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward

            # Capture more frames (every 3rd step instead of every 5th)
            if step % 3 == 0:
                frames.append(render_env.render())

            # Print key info every 10 steps
            if step % 10 == 0:
                achieved = info.get("achieved_goal", "N/A")
                desired = info.get("desired_goal", "N/A")
                if isinstance(achieved, np.ndarray) and isinstance(desired, np.ndarray):
                    dist = np.linalg.norm(achieved - desired)
                    print(f"  step {step}: distance to goal = {dist:.4f}, reward = {reward:.2f}")

            step += 1
            if terminated or truncated or step > 500:
                success = info.get("is_success", False)
                print(f"Episode ended: steps={step}, reward={episode_reward:.2f}, success={success}")

                if frames:
                    # Add 1 second pause at the end
                    pause_frame = frames[-1].copy()
                    for _ in range(12):
                        frames.append(pause_frame)

                    gif_path = os.path.join(save_dir, f"debug_ep_{ep+1}_{'success' if success else 'fail'}.gif")
                    # Slower playback at 12 fps with pause
                    imageio.mimsave(gif_path, frames, fps=12, loop=0)
                    print(f"Saved debug GIF: {gif_path}")

                if success:
                    successful_eps.append(ep+1)
                else:
                    failed_eps.append(ep+1)
                break

    print(f"\n=== Debug Summary ===")
    print(f"Successful episodes: {successful_eps} ({len(successful_eps)}/{n_episodes})")
    print(f"Failed episodes: {failed_eps} ({len(failed_eps)}/{n_episodes})")

# -------------------------
# Build SAC with HerReplayBuffer
# -------------------------
def build_sac_with_her(env, n_sampled_goal, ent_coef, learning_rate):
    """
    Build SAC model with HerReplayBuffer for stable-baselines3 2.7.0+
    """
    model = SAC(
        policy="MultiInputPolicy",
        env=env,
        learning_rate=learning_rate,
        buffer_size=Config.BUFFER_SIZE,
        batch_size=Config.BATCH_SIZE,
        tau=Config.TAU,
        gamma=Config.GAMMA,
        train_freq=Config.TRAIN_FREQ,
        gradient_steps=Config.GRADIENT_STEPS,
        learning_starts=Config.LEARNING_STARTS,
        ent_coef=ent_coef,
        use_sde=False,  # Disable state-dependent exploration for stability
        policy_kwargs=dict(
            net_arch=Config.POLICY_NET,
            n_critics=2,  # Ensure we're using twin Q-networks
        ),
        replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs=dict(
            n_sampled_goal=n_sampled_goal,
            goal_selection_strategy="future",
            online_sampling=True,
            max_episode_length=50,  # Match typical episode length
        ),
        tensorboard_log=Config.LOG_DIR,
        device=Config.DEVICE,
        verbose=1,
    )
    return model

# -------------------------
# Two-phase training with resume support
# -------------------------
def train_resume_two_phase():
    # Create environments with Monitor wrapper
    monitor_dir = os.path.join(Config.LOG_DIR, "monitor")
    env = make_env(Config.ENV_NAME, Config.REWARD_TYPE, render_mode="rgb_array", monitor_dir=monitor_dir)
    eval_env = make_env(Config.ENV_NAME, Config.REWARD_TYPE, render_mode="rgb_array")

    # Create callbacks once and reuse across phases
    eval_cb = EvalAndSaveCallback(eval_env, eval_freq=Config.EVAL_FREQ, n_eval_episodes=Config.N_EVAL_EPISODES)
    checkpoint_cb = CheckpointCallback(save_freq=Config.CHECKPOINT_FREQ, save_path=Config.MODEL_DIR, name_prefix="sac_her_checkpoint")

    # load meta if exists (to resume phases)
    meta = load_phase_meta()
    start_phase = 0
    phase_remaining_timesteps = None
    if meta:
        start_phase = meta.get("phase_idx", 0)
        phase_remaining_timesteps = meta.get("remaining_timesteps", None)
        print(f"[INFO] Found train_meta.json: resuming from phase {start_phase} with remaining timesteps {phase_remaining_timesteps}")

    # check for latest best model to resume weights
    latest_best = os.path.join(Config.MODEL_DIR, "best_model.zip")
    latest_other = find_latest_checkpoint()
    base_model_path = latest_best if os.path.exists(latest_best) else latest_other
    if base_model_path:
        print(f"[INFO] Found existing checkpoint to resume from: {base_model_path}")

    # training loop over phases
    total_done = 0
    model = None
    for idx, phase in enumerate(Config.PHASES):
        if idx < start_phase:
            # already completed earlier phase
            total_done += phase["timesteps"]
            continue

        # determine timesteps for this run (respect resume metadata)
        if idx == start_phase and phase_remaining_timesteps:
            phase_timesteps = int(phase_remaining_timesteps)
            print(f"[INFO] Resuming phase {idx} for remaining {phase_timesteps} timesteps")
        else:
            phase_timesteps = int(phase["timesteps"])

        print(f"\n=== Phase {idx} starting: timesteps={phase_timesteps:,}, n_sampled_goal={phase['n_sampled_goal']}, ent_coef={phase['ent_coef']}, lr={phase['learning_rate']} ===")

        # Build model only once, then modify hyperparameters for subsequent phases
        if model is None:
            # First phase - build new model
            model = build_sac_with_her(env, n_sampled_goal=phase["n_sampled_goal"],
                                       ent_coef=phase["ent_coef"], learning_rate=phase["learning_rate"])
            # if checkpoint exists, load weights
            if base_model_path:
                try:
                    print(f"[INFO] Loading weights from {base_model_path}...")
                    loaded_model = SAC.load(base_model_path, env=env, device=Config.DEVICE)
                    # Copy policy parameters
                    model.policy.load_state_dict(loaded_model.policy.state_dict())
                    # Copy optimizer states if possible
                    try:
                        model.policy.optimizer.load_state_dict(loaded_model.policy.optimizer.state_dict())
                    except Exception:
                        pass
                    # IMPORTANT: Copy the timestep counter too!
                    model.num_timesteps = loaded_model.num_timesteps
                    print(f"[INFO] Weights loaded successfully. Resuming from timestep {model.num_timesteps}")
                except Exception as e:
                    print(f"[WARN] Could not load checkpoint: {e}")
        else:
            # Subsequent phase - modify hyperparameters without creating new model
            print(f"[INFO] Updating hyperparameters for Phase {idx}")

            # Update learning rate
            model.learning_rate = phase["learning_rate"]
            if hasattr(model.policy, 'optimizer'):
                for param_group in model.policy.optimizer.param_groups:
                    param_group['lr'] = phase["learning_rate"]

            # Update entropy coefficient
            if phase["ent_coef"] == "auto":
                # Keep auto-tuning enabled
                pass
            else:
                model.ent_coef = phase["ent_coef"]

            # Update HER replay buffer parameters (if needed)
            if hasattr(model.replay_buffer, 'n_sampled_goal'):
                model.replay_buffer.n_sampled_goal = phase["n_sampled_goal"]
                print(f"[INFO] Updated n_sampled_goal to {phase['n_sampled_goal']}")

        # Save phase meta so if interrupted we can resume this phase
        save_phase_meta(idx, phase_timesteps)

        # Train for the phase_timesteps
        # Important: NEVER reset timesteps - maintain continuity across all phases
        try:
            model.learn(
                total_timesteps=phase_timesteps,
                callback=[eval_cb, checkpoint_cb],
                progress_bar=True,
                reset_num_timesteps=False  # Critical: maintain timestep counter across phases
            )
        except KeyboardInterrupt:
            # save model and remaining timesteps
            print("\n[INFO] KeyboardInterrupt detected - saving model and exiting gracefully.")
            model.save(os.path.join(Config.MODEL_DIR, f"interrupted_phase_{idx}_{int(time.time())}"))
            save_phase_meta(idx, phase_timesteps)
            raise

        # phase completed -> mark next phase
        save_phase_meta(idx + 1, None)
        total_done += phase_timesteps

    print(f"\n[INFO] Training complete across all phases. Total timesteps run: {total_done:,}")
    # Save final model
    final_path = os.path.join(Config.MODEL_DIR, "sac_her_final")
    model.save(final_path)
    print(f"[INFO] Final model saved to {final_path}")

    return model, env, eval_env

# -------------------------
# Entrypoint
# -------------------------
if __name__ == "__main__":
    print("=== SAC+HER PandaPickAndPlace (two-phase, resume-capable) ===")
    print(f"Using stable-baselines3 2.7.0+ with HerReplayBuffer")
    print(f"Device = {Config.DEVICE}, Logs -> {Config.LOG_DIR}")

    # Check if we should just evaluate an existing model
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--eval-only":
        print("\n=== EVALUATION MODE ===")
        model_path = os.path.join(Config.MODEL_DIR, "best_model.zip")
        if os.path.exists(model_path):
            print(f"Loading model from: {model_path}")
            eval_env = make_env(Config.ENV_NAME, Config.REWARD_TYPE, render_mode="rgb_array")
            model = SAC.load(model_path, env=eval_env, device=Config.DEVICE)

            # Create detailed visualizations
            print("\nCreating high-quality visualizations...")
            create_detailed_visualization(model, eval_env, n_episodes=5, save_dir="best_model_videos")

            # Run evaluation
            print("\nRunning evaluation...")
            mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=20, render=False)

            success_count = 0
            for i in range(20):
                obs, _ = eval_env.reset()
                done = False
                while not done:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = eval_env.step(action)
                    done = terminated or truncated
                if info.get("is_success", False):
                    success_count += 1

            success_rate = success_count / 20
            print(f"\nResults: mean_reward={mean_reward:.3f}±{std_reward:.3f}, success_rate={success_rate*100:.1f}%")
            eval_env.close()
            sys.exit(0)
        else:
            print(f"ERROR: No model found at {model_path}")
            sys.exit(1)

    # Normal training mode
    model, train_env, eval_env = train_resume_two_phase()

    # optional debug rollouts
    print("\nRunning debug rollouts (3 episodes) to inspect failures and produce gifs...")
    debug_rollouts(model, eval_env, n_episodes=3, save_dir="debug_renders")

    # final evaluation
    mean_reward, success_rate = None, None
    try:
        # evaluate_policy returns mean & std; we compute deterministic success separately
        mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=20, render=False)
        success_count = 0
        for _ in range(20):
            obs, _ = eval_env.reset()
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = eval_env.step(action)
                done = terminated or truncated
            if info.get("is_success", False):
                success_count += 1
        success_rate = success_count / 20
        print(f"\nFinal eval: mean_reward={mean_reward:.3f}, std={std_reward:.3f}, success_rate={success_rate*100:.1f}%")
    except Exception as e:
        print(f"[WARN] final evaluation failed: {e}")

    # cleanup
    train_env.close()
    eval_env.close()
    print("Done.")
