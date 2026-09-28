"""Open the cart-pole model, independent of the viewer's working directory."""
from pathlib import Path

import mujoco
import mujoco.viewer


# Resolve before the GUI starts: the macOS window system can change cwd.
MODEL_PATH = Path(__file__).resolve().with_name("cart_pole.xml")


def load_model():
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    print(f"Loaded {MODEL_PATH} ({model.nbody} bodies, {model.njnt} joints)", flush=True)
    return model, data


if __name__ == "__main__":
    mujoco.viewer.launch(loader=load_model)
