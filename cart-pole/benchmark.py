"""Benchmark MuJoCo simulation speed on this machine."""
import time
import mujoco
from pathlib import Path

MODEL_PATH = Path(__file__).parent / "cart_pole.xml"

model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
data = mujoco.MjData(model)

# Warm up
for _ in range(1000):
    mujoco.mj_step(model, data)

# Benchmark
n_steps = 100_000
mujoco.mj_resetData(model, data)

start = time.perf_counter()
for _ in range(n_steps):
    mujoco.mj_step(model, data)
elapsed = time.perf_counter() - start

steps_per_sec = n_steps / elapsed
realtime_factor = steps_per_sec * model.opt.timestep  # how many seconds of sim per second of wall time

print(f"Model: cart_pole.xml ({model.nbody} bodies, {model.njnt} joints)")
print(f"Timestep: {model.opt.timestep * 1000:.1f} ms")
print(f"")
print(f"Steps computed: {n_steps:,}")
print(f"Wall time: {elapsed:.2f} s")
print(f"")
print(f"Steps/second: {steps_per_sec:,.0f}")
print(f"Realtime factor: {realtime_factor:,.0f}x")
print(f"")
print(f"This means: {realtime_factor:.0f} seconds of simulated physics per 1 second of real time")
