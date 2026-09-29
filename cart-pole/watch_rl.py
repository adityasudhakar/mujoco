"""Watch a trained RL policy balance the cart-pole.

Usage:
    python watch_rl.py checkpoints/cartpole_ppo_5000_steps.zip   # early (bad)
    python watch_rl.py checkpoints/cartpole_ppo_final.zip        # final (good)
"""
import sys
import mujoco
import mujoco.viewer
from pathlib import Path
from stable_baselines3 import PPO
import numpy as np

MODEL_PATH = Path(__file__).parent / "cart_pole.xml"


class RLController:
    """Controller that uses a trained RL policy."""

    def __init__(self, model, data, policy_path):
        self.model = model
        self.data = data

        # Load trained policy
        print(f"Loading policy from: {policy_path}")
        self.policy = PPO.load(policy_path)

        # Find joint indices
        cart_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "cart_slide")
        pole_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "pole_hinge")
        self.cart_actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "cart_motor")

        # Joint addresses
        self.cart_qpos_addr = model.jnt_qposadr[cart_joint_id]
        self.cart_qvel_addr = model.jnt_dofadr[cart_joint_id]
        self.pole_qpos_addr = model.jnt_qposadr[pole_joint_id]
        self.pole_qvel_addr = model.jnt_dofadr[pole_joint_id]

        # Set initial tilt
        data.qpos[self.pole_qpos_addr] = 0.1
        mujoco.mj_forward(model, data)

        # Frame skip to match training (run policy every 20 physics steps)
        self.frame_skip = 20
        self.step_counter = 0

        print()
        print("RL Policy Controller")
        print("====================")
        print("Try pushing the pole with Ctrl + drag!")
        print()

    def _get_obs(self, data):
        """Get current observation from the given data object."""
        return np.array([
            data.qpos[self.cart_qpos_addr],
            data.qvel[self.cart_qvel_addr],
            data.qpos[self.pole_qpos_addr],
            data.qvel[self.pole_qvel_addr],
        ], dtype=np.float32)

    def __call__(self, model, data):
        """Called at each timestep by the viewer."""
        self.step_counter += 1

        # Only run policy every frame_skip steps (to match training frequency)
        if self.step_counter % self.frame_skip != 0:
            return  # Keep applying previous control

        obs = self._get_obs(data)
        pole_angle = obs[2]
        cart_pos = obs[0]

        # Reset if fallen or out of bounds (like during training)
        if abs(pole_angle) > 0.5 or abs(cart_pos) > 2.0:
            mujoco.mj_resetData(model, data)
            data.qpos[self.pole_qpos_addr] = 0.1  # small initial tilt
            mujoco.mj_forward(model, data)
            self.step_counter = 0
            return

        # Get action from policy (deterministic for visualization)
        action, _ = self.policy.predict(obs, deterministic=True)

        # Scale action to force (-1,1) -> (-100, 100)
        force = float(action[0]) * 100.0

        # Apply control
        data.ctrl[self.cart_actuator_id] = force


def load_and_setup(policy_path):
    """Load model and set up RL controller."""
    def loader():
        model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
        data = mujoco.MjData(model)

        controller = RLController(model, data, policy_path)
        mujoco.set_mjcb_control(controller)

        return model, data
    return loader


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python watch_rl.py <path_to_policy.zip>")
        print()
        print("Example:")
        print("  python watch_rl.py checkpoints/cartpole_ppo_5000_steps.zip")
        print("  python watch_rl.py checkpoints/cartpole_ppo_final.zip")
        sys.exit(1)

    policy_path = sys.argv[1]

    if not Path(policy_path).exists():
        print(f"Error: Policy file not found: {policy_path}")
        sys.exit(1)

    mujoco.viewer.launch(loader=load_and_setup(policy_path))
