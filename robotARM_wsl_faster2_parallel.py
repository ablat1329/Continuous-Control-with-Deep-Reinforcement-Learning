import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"      # disables oneDNN optimizations
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"       # hides all TF info/warning logs
import multiprocessing
import imageio
import gymnasium as gym
import panda_gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize, SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy


# ==============================================
#  DISPLAY CONFIG (for WSL users only)
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
#  CALLBACK: periodic evaluation + GIF creation
# ==============================================
class EvalGifCallback(BaseCallback):
    def __init__(self, eval_env, eval_freq=100000, n_eval_episodes=3, save_dir="renders", verbose=1):
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
            print(f"\n[INFO] Step {self.num_timesteps:,}: mean_reward={mean_reward:.2f} ± {std_reward:.2f}")

            frames = []
            obs, _ = self.eval_env.reset()
            for i in range(400):
                action, _ = self.model.predict(obs)
                obs, reward, done, truncated, info = self.eval_env.step(action)
                if i % 10 == 0:
                    frame = self.eval_env.render()
                    frames.append(frame)
                if done or truncated:
                    obs, _ = self.eval_env.reset()

            gif_path = os.path.join(self.save_dir, f"eval_step_{self.num_timesteps}.gif")
            imageio.mimsave(gif_path, frames, fps=10)
            print(f"[INFO] Saved evaluation GIF: {gif_path}\n")
        return True


# ==============================================
#  MAIN ENTRY POINT (required for Windows)
# ==============================================
if __name__ == "__main__":
    # Use all cores minus one
    NUM_ENVS = max(2, multiprocessing.cpu_count() - 1)
    env_kwargs = {"reward_type": "dense", "render_mode": "rgb_array"}

    print(f"[INFO] Using {NUM_ENVS} parallel environments for training.")

    # Create SubprocVecEnv (true multiprocessing)
    train_env = make_vec_env(
        "PandaPickAndPlace-v3",
        n_envs=NUM_ENVS,
        env_kwargs=env_kwargs,
        vec_env_cls=SubprocVecEnv,
    )

    # Normalize observations & rewards
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    # Evaluation environment
    eval_env = gym.make("PandaPickAndPlace-v3", **env_kwargs)

    # PPO setup
    policy_kwargs = dict(log_std_init=-2, ortho_init=False)

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
        learning_rate=1e-4,
        clip_range=0.1,
        max_grad_norm=0.5,
        policy_kwargs=policy_kwargs,
        tensorboard_log="./logs/",
        device="auto"
    )

    callback = EvalGifCallback(eval_env, eval_freq=100000, save_dir="renders")

    print("\n🚀 Starting PPO training (multi-core)...\n")
    model.learn(total_timesteps=10_000_000, callback=callback)
    print("\n✅ Training complete.\n")

    # Save VecNormalize stats before evaluation
    train_env.save("vec_normalize.pkl")

    # Reload normalization for evaluation
    eval_env = VecNormalize.load("vec_normalize.pkl", eval_env)
    eval_env.training = False
    eval_env.norm_reward = False

    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10)
    print(f"🏁 Final Evaluation → mean_reward={mean_reward:.2f} ± {std_reward:.2f}")

    model.save("ppo_panda_pickplace_multi")
    print("💾 Model and VecNormalize stats saved.")

    train_env.close()
    eval_env.close()
