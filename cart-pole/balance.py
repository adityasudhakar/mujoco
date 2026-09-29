"""Cart-pole with PD balancing controller.

On macOS, run with mjpython:
    mjpython balance.py

Or if mjpython isn't in PATH:
    /path/to/.venv312/bin/mjpython balance.py
"""
import mujoco
import mujoco.viewer
from pathlib import Path

MODEL_PATH = Path(__file__).parent / "cart_pole.xml"

# Controller gains - tune these!
Kp = 300.0   # Proportional gain (react to angle)
Kd = 50.0    # Derivative gain (react to velocity)

# Initial tilt so we can watch it correct
INITIAL_TILT = 0.1  # radians (~6 degrees)


class CartPoleController:
    def __init__(self, model, data):
        self.model = model
        self.data = data

        # Find joint and actuator indices
        self.pole_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "pole_hinge")
        self.cart_actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "cart_motor")

        # Get the qpos/qvel address for the pole joint
        self.pole_qpos_addr = model.jnt_qposadr[self.pole_joint_id]
        self.pole_qvel_addr = model.jnt_dofadr[self.pole_joint_id]

        # Set initial tilt
        data.qpos[self.pole_qpos_addr] = INITIAL_TILT
        mujoco.mj_forward(model, data)

        print(f"Cart-pole PD Controller")
        print(f"=======================")
        print(f"Kp = {Kp}, Kd = {Kd}")
        print(f"Initial tilt: {INITIAL_TILT:.2f} rad ({INITIAL_TILT * 57.3:.1f}°)")
        print()
        print("Try pushing the pole with Ctrl + drag!")
        print()

    def __call__(self, model, data):
        """Called at each timestep by the viewer."""
        # Read pole state
        pole_angle = data.qpos[self.pole_qpos_addr]
        pole_velocity = data.qvel[self.pole_qvel_addr]

        # PD control (positive: if pole leans right, push cart right to get under it)
        force = Kp * pole_angle + Kd * pole_velocity

        # Clamp to actuator limits (-100 to 100 N)
        force = max(-100, min(100, force))

        # Apply control
        data.ctrl[self.cart_actuator_id] = force


def load_and_setup():
    """Load model and set up controller."""
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)

    controller = CartPoleController(model, data)

    # Register controller as physics callback
    mujoco.set_mjcb_control(controller)

    return model, data


if __name__ == "__main__":
    mujoco.viewer.launch(loader=load_and_setup)
