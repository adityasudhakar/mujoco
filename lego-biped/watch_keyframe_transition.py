#!/usr/bin/env python3
"""Watch the biped get from the bad prone pose to a sustained full stand."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import mujoco
from mujoco import viewer as mj_viewer
from stable_baselines3 import PPO

from biped_env import CONTROL_DT
from keyframe_transition_env import ProneKeyframeTransitionEnv


DEFAULT_MODEL = (
    Path(__file__).with_name("checkpoints")
    / "prone_keyframe_transition"
    / "transition_ppo_final.zip"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", nargs="?", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--stochastic", action="store_true")
    args = parser.parse_args()

    policy = PPO.load(args.model)
    env = ProneKeyframeTransitionEnv()
    observation, _ = env.reset()
    state = {"paused": False, "reset": False}
    episode_sway = 0.0

    def on_key(keycode: int) -> None:
        if keycode in (ord("P"), ord("p")):
            state["paused"] = not state["paused"]
        elif keycode in (ord("R"), ord("r")):
            state["reset"] = True

    with mj_viewer.launch_passive(
        env.model, env.data, key_callback=on_key
    ) as viewer:
        viewer.cam.lookat[:] = [0.0, 0.0, 0.18]
        viewer.cam.distance = 1.15
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -18
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = 1
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = 1

        while viewer.is_running():
            if state["reset"]:
                observation, _ = env.reset()
                episode_sway = 0.0
                state["reset"] = False
            if state["paused"]:
                viewer.sync()
                time.sleep(CONTROL_DT)
                continue

            started = time.perf_counter()
            action, _ = policy.predict(
                observation, deterministic=not args.stochastic
            )
            observation, _, terminated, truncated, info = env.step(action)
            episode_sway += info["foot_sway"]
            viewer.sync()
            if terminated or truncated:
                result = "STAND COMPLETE" if info["is_success"] else "STAND FAILED"
                print(
                    f"{result}: distance={info['goal_distance']:.3f} "
                    f"z={info['torso_z']:.3f} upright={info['upright']:+.3f} "
                    f"mean_planted_foot_sway="
                    f"{episode_sway / max(1, env.step_count):.5f} m/s",
                    flush=True,
                )
                time.sleep(1.0)
                observation, _ = env.reset()
                episode_sway = 0.0

            remaining = CONTROL_DT - (time.perf_counter() - started)
            if remaining > 0:
                time.sleep(remaining)

    env.close()


if __name__ == "__main__":
    main()
