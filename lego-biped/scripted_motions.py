#!/usr/bin/env python3
"""Play the LEGO biped's pre-RL scripted motions.

These are deliberately transparent keyframe controllers, not learned policies.
Each tuple is (duration_seconds, [left hip, left knee, right hip, right knee])
in degrees.  Targets are interpolated so the servos do not receive impossible
instantaneous jumps.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import mujoco
import numpy as np


MODEL_PATH = Path(__file__).with_name("lego_biped.xml")
CONTROL_HZ = 50.0
CONTROL_PERIOD = 1.0 / CONTROL_HZ


MOTIONS_DEG: dict[str, list[tuple[float, list[float]]]] = {
    # Hold the XML's named standing keyframe for inspection.
    "stand": [
        (0.5, [0, 0, 0, 0]),
    ],
    # Incremental prone-development test.  This deliberately does nothing:
    # it lets us verify the flat reset pose before moving any actuator.
    "prone_flat": [
        (0.5, [0, 0, 0, 0]),
    ],
    # A slow, exaggerated planar gait.  The crouch creates foot clearance;
    # alternate hip motion advances one foot while the other supports the body.
    "walk": [
        (0.8, [0, -25, 0, -25]),
        (0.35, [-18, -65, 12, -12]),
        (0.35, [20, -18, -12, -30]),
        (0.35, [12, -12, -18, -65]),
        (0.35, [-12, -30, 20, -18]),
    ] * 4 + [(0.8, [0, 0, 0, 0])],

    # Fold both legs under the torso, use the feet as a contact point, then
    # extend in stages rather than attempting one abrupt push.
    "getup_prone_1": [
        (0.5, [-180, -120, -180, -120]),
        (1.0, [45, 0, 45, 0]),
        (0.5, [-20, 0, -20, 0]),
        (0.5, [0, 0, 0, 0]),
    ],

    # Fold both legs under the torso, use the feet as a contact point, then
    # extend in stages rather than attempting one abrupt push.
    "getup_prone_2": [
        (0.01, [-55, -100, -55, -100]),
        # The feet have planted.  Reverse the knees now instead of curling
        # farther: hold the hips and extend the lower legs to lift the body.
        (1.0, [-55, -60, -55, -60]),
        (1.0, [-55, -25, -55, -25]),
        # Once the legs are mostly extended, rotate the hips to bring the
        # torso over the feet, then finish straightening.
        (1.0, [-25, -25, -25, -25]),
        (1.0, [0, -25, 0, -25]),
        (0.8, [0, -10, 0, -10]),
        (0.8, [0, 0, 0, 0]),
    ],

    # Curl the feet toward the body, plant them, then straighten the knees and
    # hips together to lever the torso upward.
    "getup_supine": [
        (0.8, [70, -105, 70, -105]),
        (0.8, [110, -135, 110, -135]),
        (0.8, [80, -95, 80, -95]),
        (0.9, [45, -55, 45, -55]),
        (1.0, [18, -22, 18, -22]),
        (1.0, [0, 0, 0, 0]),
    ],
}


RESET_KEY = {
    "stand": "stand",
    "prone_flat": "prone",
    "walk": "stand",
    "getup_prone_1": "prone",
    "getup_prone_2": "prone",
    "getup_supine": "supine",
}


def key_id(model: mujoco.MjModel, name: str) -> int:
    idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, name)
    if idx < 0:
        raise ValueError(f"missing keyframe {name!r}")
    return idx


def reset_for_motion(model: mujoco.MjModel, data: mujoco.MjData, motion: str) -> None:
    mujoco.mj_resetDataKeyframe(model, data, key_id(model, RESET_KEY[motion]))
    mujoco.mj_forward(model, data)


def torso_metrics(model: mujoco.MjModel, data: mujoco.MjData) -> tuple[float, float, float]:
    torso_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso")
    x = float(data.xpos[torso_id, 0])
    z = float(data.xpos[torso_id, 2])
    # Torso local +Z projected onto world +Z: 1 upright, 0 horizontal, -1 inverted.
    upright = float(data.xmat[torso_id].reshape(3, 3)[2, 2])
    return x, z, upright


def passed(motion: str, x: float, z: float, upright: float) -> bool:
    standing = z > 0.28 and upright > 0.90
    if motion == "walk":
        return standing and x > 0.20
    return standing


def run_motion(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    motion: str,
    viewer=None,
    realtime: bool = True,
    playback_state: dict[str, bool] | None = None,
) -> None:
    reset_for_motion(model, data, motion)
    previous = data.ctrl.copy()
    physics_steps_per_control = max(1, round(CONTROL_PERIOD / model.opt.timestep))

    for duration, target_deg in MOTIONS_DEG[motion]:
        target = np.deg2rad(np.asarray(target_deg, dtype=np.float64))
        ticks = max(1, round(duration * CONTROL_HZ))
        segment_start = previous.copy()

        for tick in range(ticks):
            if playback_state is not None:
                while playback_state["paused"]:
                    if playback_state["replay"] or viewer is None or not viewer.is_running():
                        return
                    viewer.sync()
                    time.sleep(CONTROL_PERIOD)
                if playback_state["replay"]:
                    return

            blend = (tick + 1) / ticks
            # Smoothstep interpolation gives zero slope at each keyframe.
            blend = blend * blend * (3.0 - 2.0 * blend)
            data.ctrl[:] = segment_start + blend * (target - segment_start)

            wall_start = time.perf_counter()
            for _ in range(physics_steps_per_control):
                mujoco.mj_step(model, data)

            if viewer is not None:
                viewer.sync()
            if realtime:
                remaining = CONTROL_PERIOD - (time.perf_counter() - wall_start)
                if remaining > 0:
                    time.sleep(remaining)

        previous = target

    # Hold the final pose briefly so it is easy to inspect.
    for _ in range(round(1.5 * CONTROL_HZ)):
        wall_start = time.perf_counter()
        for _ in range(physics_steps_per_control):
            mujoco.mj_step(model, data)
        if viewer is not None:
            viewer.sync()
        if realtime:
            remaining = CONTROL_PERIOD - (time.perf_counter() - wall_start)
            if remaining > 0:
                time.sleep(remaining)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("motion", choices=sorted(MOTIONS_DEG))
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)

    if args.headless:
        run_motion(model, data, args.motion, realtime=False)
    else:
        from mujoco import viewer as mj_viewer

        playback_state = {"paused": False, "replay": False}

        def on_key(keycode: int) -> None:
            if keycode in (ord("R"), ord("r")):
                playback_state["replay"] = True
                playback_state["paused"] = False
                print("REPLAY", flush=True)
            elif keycode in (ord("P"), ord("p")):
                playback_state["paused"] = not playback_state["paused"]
                label = "PAUSED" if playback_state["paused"] else "RUNNING"
                targets = np.rad2deg(data.ctrl)
                print(
                    f"{label} targets_deg="
                    f"[{targets[0]:+.1f}, {targets[1]:+.1f}, "
                    f"{targets[2]:+.1f}, {targets[3]:+.1f}]",
                    flush=True,
                )

        with mj_viewer.launch_passive(model, data, key_callback=on_key) as viewer:
            viewer.cam.lookat[:] = [0.0, 0.0, 0.18]
            viewer.cam.distance = 1.15
            viewer.cam.azimuth = 135
            viewer.cam.elevation = -18
            while viewer.is_running():
                playback_state["replay"] = False
                playback_state["paused"] = False
                # Show the named starting pose before each playback.
                reset_for_motion(model, data, args.motion)
                viewer.sync()
                time.sleep(1.5)
                run_motion(
                    model,
                    data,
                    args.motion,
                    viewer=viewer,
                    playback_state=playback_state,
                )

                # launch_passive does not advance physics for us.  Keep
                # stepping after playback, and replay when the user presses R.
                physics_steps = max(1, round(CONTROL_PERIOD / model.opt.timestep))
                while viewer.is_running() and not playback_state["replay"]:
                    if playback_state["paused"]:
                        viewer.sync()
                        time.sleep(CONTROL_PERIOD)
                        continue
                    wall_start = time.perf_counter()
                    for _ in range(physics_steps):
                        mujoco.mj_step(model, data)
                    viewer.sync()
                    remaining = CONTROL_PERIOD - (time.perf_counter() - wall_start)
                    if remaining > 0:
                        time.sleep(remaining)

    x, z, upright = torso_metrics(model, data)
    status = "PASS" if passed(args.motion, x, z, upright) else "NOT YET"
    print(
        f"motion={args.motion} final_x={x:+.3f}m final_z={z:.3f}m "
        f"upright={upright:+.3f} status={status}"
    )


if __name__ == "__main__":
    main()
