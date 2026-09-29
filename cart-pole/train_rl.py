"""Train an RL agent to balance the cart-pole.

This creates a Gymnasium environment wrapper around our MuJoCo cart-pole,
then trains a PPO agent to balance it.

Saves checkpoints at different stages so you can compare early (bad) vs late (good) policies.
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
import time

MODEL_PATH = Path(__file__).parent / "cart_pole.xml"
CHECKPOINT_DIR = Path(__file__).parent / "checkpoints"


class CartPoleEnv(gym.Env):
    """Gymnasium environment for our MuJoCo cart-pole."""

    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode=None):
        super().__init__()

        # Load MuJoCo model
        self.mj_model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
        self.mj_data = mujoco.MjData(self.mj_model)

        # Find joint indices
        self.cart_joint_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, "cart_slide")
        self.pole_joint_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, "pole_hinge")
        self.cart_actuator_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, "cart_motor")

        # Joint addresses
        self.cart_qpos_addr = self.mj_model.jnt_qposadr[self.cart_joint_id]
        self.cart_qvel_addr = self.mj_model.jnt_dofadr[self.cart_joint_id]
        self.pole_qpos_addr = self.mj_model.jnt_qposadr[self.pole_joint_id]
        self.pole_qvel_addr = self.mj_model.jnt_dofadr[self.pole_joint_id]

        # Action space: force on cart, normalized to [-1, 1]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        # Observation space: [cart_pos, cart_vel, pole_angle, pole_vel]
        # Using generous bounds
        high = np.array([2.5, 10.0, np.pi, 10.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

        # Motor force scaling (action in [-1,1] maps to [-100, 100] N)
        self.force_scale = 100.0

        # Frame skip to match Gymnasium (20 physics steps per action = 0.04s)
        self.frame_skip = 20

        # Episode settings
        self.max_steps = 500  # Now ~20 seconds at 25 Hz control
        self.step_count = 0

        self.render_mode = render_mode

    def _get_obs(self):
        """Get current observation."""
        return np.array([
            self.mj_data.qpos[self.cart_qpos_addr],   # cart position
            self.mj_data.qvel[self.cart_qvel_addr],   # cart velocity
            self.mj_data.qpos[self.pole_qpos_addr],   # pole angle
            self.mj_data.qvel[self.pole_qvel_addr],   # pole angular velocity
        ], dtype=np.float32)

    def reset(self, seed=None, options=None):
        """Reset environment to initial state with small random perturbation."""
        super().reset(seed=seed)

        mujoco.mj_resetData(self.mj_model, self.mj_data)

        # Small random initial tilt (so it has something to correct)
        self.mj_data.qpos[self.pole_qpos_addr] = self.np_random.uniform(-0.1, 0.1)
        self.mj_data.qvel[self.pole_qvel_addr] = self.np_random.uniform(-0.1, 0.1)

        mujoco.mj_forward(self.mj_model, self.mj_data)

        self.step_count = 0

        return self._get_obs(), {}

    def step(self, action):
        """Take one step in the environment."""
        # Apply action (scale from [-1,1] to [-100,100])
        force = float(action[0]) * self.force_scale
        self.mj_data.ctrl[self.cart_actuator_id] = force

        # Step physics multiple times (frame skip)
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.mj_model, self.mj_data)

        self.step_count += 1

        # Get new state
        obs = self._get_obs()
        cart_pos, cart_vel, pole_angle, pole_vel = obs

        # Check termination conditions
        fallen = abs(pole_angle) > 0.5  # ~29 degrees
        out_of_bounds = abs(cart_pos) > 2.0
        timeout = self.step_count >= self.max_steps

        terminated = fallen or out_of_bounds
        truncated = timeout and not terminated

        # Simple reward: +1 per step survived (like Gymnasium)
        # No complex shaping - let the agent learn purely from survival
        reward = 1.0

        return obs, reward, terminated, truncated, {}

    def close(self):
        pass


def train():
    """Train the RL agent with checkpoints."""
    print("=" * 50)
    print("Training RL agent for cart-pole balancing")
    print("=" * 50)
    print()

    # Create environment
    env = CartPoleEnv()

    # Create checkpoint directory
    CHECKPOINT_DIR.mkdir(exist_ok=True)

    # Checkpoint callback - save every 50000 steps
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path=str(CHECKPOINT_DIR),
        name_prefix="cartpole_ppo",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    # Create PPO agent
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        device="cpu",  # Use CPU for simplicity
    )

    print("Training for 500,000 timesteps...")
    print("Checkpoints saved every 50,000 steps to:", CHECKPOINT_DIR)
    print()

    start = time.time()
    model.learn(total_timesteps=500000, callback=checkpoint_callback)
    elapsed = time.time() - start

    # Save final model
    final_path = CHECKPOINT_DIR / "cartpole_ppo_final"
    model.save(str(final_path))

    print()
    print("=" * 50)
    print(f"Training complete in {elapsed:.1f} seconds")
    print(f"Final model saved to: {final_path}.zip")
    print()
    print("Checkpoints saved (earliest to latest):")
    for f in sorted(CHECKPOINT_DIR.glob("cartpole_ppo_*.zip")):
        print(f"  {f.name}")
    print()
    print("To watch a policy, run:")
    print("  python watch_rl.py checkpoints/cartpole_ppo_5000_steps.zip   # early (bad)")
    print("  python watch_rl.py checkpoints/cartpole_ppo_final.zip        # final (good)")
    print("=" * 50)

    env.close()


if __name__ == "__main__":
    train()
