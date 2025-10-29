"""
Script to load existing SAC+HER model and continue training with higher entropy
"""
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"      # disables oneDNN optimizations
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"       # hides all TF info/warning logs

import gymnasium as gym
import panda_gym  # Required to register Panda environments
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
import sys

# Add your callback if you want to keep using it
# from your_original_script import EnhancedEvalCallback


def find_latest_checkpoint(models_dir="models"):
    """Find the most recent checkpoint"""
    if not os.path.exists(models_dir):
        print(f"❌ Directory {models_dir} not found!")
        return None

    checkpoints = [f for f in os.listdir(models_dir) if f.startswith("sac_her_checkpoint")]
    if not checkpoints:
        print(f"❌ No checkpoints found in {models_dir}")
        return None

    # Sort by timesteps in filename
    checkpoints.sort(key=lambda x: int(x.split('_')[-2]) if x.split('_')[-2].isdigit() else 0)
    latest = checkpoints[-1]

    checkpoint_path = os.path.join(models_dir, latest.replace('.zip', ''))
    print(f"✅ Found checkpoint: {latest}")
    return checkpoint_path


def continue_training_with_new_entropy(
    checkpoint_path=None,
    new_entropy=0.5,
    additional_timesteps=2_000_000,
    models_dir="models",
    log_dir="./tensorboard_logs/"
):
    """
    Load existing model and continue training with modified entropy

    Args:
        checkpoint_path: Path to checkpoint (if None, finds latest)
        new_entropy: New entropy coefficient (0.5 for more exploration)
        additional_timesteps: How many more steps to train
        models_dir: Directory containing checkpoints
        log_dir: TensorBoard log directory
    """

    # Find checkpoint if not specified
    if checkpoint_path is None:
        checkpoint_path = find_latest_checkpoint(models_dir)
        if checkpoint_path is None:
            return

    print(f"\n{'='*60}")
    print("🔄 CONTINUING TRAINING WITH MODIFIED ENTROPY")
    print(f"{'='*60}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"New Entropy: {new_entropy}")
    print(f"Additional Steps: {additional_timesteps:,}")
    print(f"{'='*60}\n")

    # Create environment (must match training env)
    env_kwargs = {
        "reward_type": "sparse",  # Match your training config
        "render_mode": "rgb_array"
    }
    train_env = gym.make("PandaPickAndPlace-v3", **env_kwargs)
    eval_env = gym.make("PandaPickAndPlace-v3", **env_kwargs)

    # Load the model
    print("📂 Loading model...")
    try:
        model = SAC.load(checkpoint_path, env=train_env)
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    # Get current training stats
    current_timesteps = model.num_timesteps
    print(f"\n📊 Current Progress: {current_timesteps:,} timesteps")

    # Modify entropy coefficient
    print(f"\n🔧 Modifying entropy coefficient...")
    print(f"   Old entropy: {model.ent_coef}")

    # Set new entropy
    model.ent_coef = new_entropy
    if hasattr(model, 'ent_coef_tensor'):
        model.ent_coef_tensor = torch.tensor([new_entropy]).to(model.device)

    print(f"   New entropy: {model.ent_coef}")
    print(f"   ✅ Entropy updated! Robot will explore more.\n")

    # Setup callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=100_000,
        save_path=models_dir,
        name_prefix="sac_her_continued"
    )

    # Optional: Add evaluation callback if you have it
    # eval_callback = EnhancedEvalCallback(...)

    # Continue training
    print(f"🚀 Resuming training for {additional_timesteps:,} more steps...\n")
    try:
        model.learn(
            total_timesteps=additional_timesteps,
            callback=checkpoint_callback,
            reset_num_timesteps=True,  # Keep counting from current timesteps
            progress_bar=True,
            tb_log_name="SAC_continued",
            log_interval=10
        )
        print("\n✅ Training completed!")
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during training: {e}")

    # Save final model
    final_path = os.path.join(models_dir, "sac_her_continued_final")
    model.save(final_path)
    print(f"\n💾 Final model saved to: {final_path}")

    # Cleanup
    train_env.close()
    eval_env.close()

    print(f"\n{'='*60}")
    print(f"Total timesteps: {model.num_timesteps:,}")
    print(f"{'='*60}")


def save_current_model_from_keyboard_interrupt(models_dir="models"):
    """
    If your training is currently running and you want to save it:
    1. Press Ctrl+C to interrupt
    2. The model should auto-save via callback
    3. Or manually save if needed
    """
    print("""
    ⚠️  TO SAVE YOUR CURRENTLY RUNNING MODEL:

    1. Press Ctrl+C to interrupt training gracefully
       - Your callback should auto-save the model

    2. The model will be saved in your 'models/' directory as:
       - sac_her_checkpoint_XXXXXX_steps.zip

    3. Then run this script to continue with higher entropy!
    """)


if __name__ == "__main__":
    print("""
    ╔════════════════════════════════════════════════════════════╗
    ║  SAC+HER TRAINING CONTINUATION SCRIPT                      ║
    ╚════════════════════════════════════════════════════════════╝

    This script will:
    1. Load your latest checkpoint
    2. Increase entropy coefficient to 0.5 (more exploration)
    3. Continue training for 2M more steps

    """)

    # Check if models directory exists
    if not os.path.exists("models"):
        print("❌ 'models' directory not found!")
        print("\nYour training script should have created checkpoints in 'models/' directory.")
        print("If your training is currently running:")
        save_current_model_from_keyboard_interrupt()
        sys.exit(1)

    # Ask user for confirmation
    response = input("Continue training with entropy=0.5? (yes/no): ").lower()

    if response in ['yes', 'y']:
        # You can modify these parameters:
        continue_training_with_new_entropy(
            checkpoint_path=None,  # Auto-find latest
            new_entropy=0.5,       # Increase from 0.2 to 0.5
            additional_timesteps=2_000_000,  # Train 2M more steps
        )
    else:
        print("❌ Cancelled")
        print("\nTo manually specify a checkpoint:")
        print("continue_training_with_new_entropy(")
        print("    checkpoint_path='models/sac_her_checkpoint_500000_steps',")
        print("    new_entropy=0.5,")
        print("    additional_timesteps=2_000_000")
        print(")")
