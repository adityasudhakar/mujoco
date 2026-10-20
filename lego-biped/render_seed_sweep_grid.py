#!/usr/bin/env python3
"""Render a 5x5 video grid of successful and failed prone-get-up policies."""

from __future__ import annotations

import csv
import io
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from stable_baselines3 import PPO

from keyframe_transition_env import ProneKeyframeTransitionEnv


HERE = Path(__file__).resolve().parent
CHECKPOINTS = HERE / "checkpoints"
SWEEP_DIR = CHECKPOINTS / "seed_sweep_8_100"
ASSETS = HERE / "assets"
SUMMARY = SWEEP_DIR / "summary.csv"
OUTPUT = ASSETS / "seed-sweep-5x5.mp4"
POSTER = ASSETS / "seed-sweep-5x5-poster.jpg"

SUCCESS_SEEDS = (59, 54, 93, 91, 17, 68, 46, 44, 21, 39, 64)
# Representative failed runs spanning high reward, low reward, and especially
# low sway despite failing the combined stand criterion.
FAILURE_SEEDS = (81, 60, 82, 43, 56, 99, 40, 37, 63, 29, 30, 75, 88)
BC_MODEL = CHECKPOINTS / "prone_height_sway_balanced_10_seed6" / "best" / "best_model.zip"

GRID_COLUMNS = 5
GRID_ROWS = 5
PANEL_WIDTH = 320
PANEL_HEIGHT = 240
HEADER_HEIGHT = 38
FOOTER_HEIGHT = 44
FPS = 25
FRAMES = 150  # 6 seconds; render every other 50 Hz control step.
POSTER_FRAME = 115


@dataclass(frozen=True)
class PolicyPanel:
    title: str
    subtitle: str
    model_path: Path
    kind: str


def load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = (
        ["/System/Library/Fonts/Supplemental/Arial Bold.ttf",
         "/Library/Fonts/Arial Bold.ttf"]
        if bold
        else ["/System/Library/Fonts/Supplemental/Arial.ttf",
              "/Library/Fonts/Arial.ttf"]
    )
    for path in candidates:
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            pass
    return ImageFont.load_default()


def camera_for_scene() -> mujoco.MjvCamera:
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.lookat[:] = [-0.06, 0.0, 0.17]
    camera.distance = 1.0
    camera.azimuth = 135
    camera.elevation = -18
    return camera


def panels_from_summary() -> list[PolicyPanel]:
    with SUMMARY.open(newline="", encoding="utf-8") as summary_file:
        rows = {int(row["seed"]): row for row in csv.DictReader(summary_file)}

    panels: list[PolicyPanel] = []
    for seed in SUCCESS_SEEDS:
        row = rows[seed]
        if row["successes_out_of_20"] != "20":
            raise ValueError(f"seed {seed} no longer has a 20/20 result")
        panels.append(PolicyPanel(
            title=f"SEED {seed}  |  PASS 20/20",
            subtitle=f"mean planted-foot sway {float(row['mean_planted_foot_sway_m_s']):.3f} m/s",
            model_path=CHECKPOINTS
            / f"prone_height_sway_balanced_10_seed{seed}_sweep"
            / "height_sway_ppo_final.zip",
            kind="success",
        ))

    for seed in FAILURE_SEEDS:
        row = rows[seed]
        if row["successes_out_of_20"] != "0":
            raise ValueError(f"seed {seed} is not a failed example")
        panels.append(PolicyPanel(
            title=f"SEED {seed}  |  FAIL 0/20",
            subtitle=f"mean planted-foot sway {float(row['mean_planted_foot_sway_m_s']):.3f} m/s",
            model_path=CHECKPOINTS
            / f"prone_height_sway_balanced_10_seed{seed}_sweep"
            / "height_sway_ppo_final.zip",
            kind="failure",
        ))

    panels.append(PolicyPanel(
        title="BC SEED 6  |  PASS 10/10 PRE-PPO",
        subtitle="script-imitation baseline, before PPO fine-tuning",
        model_path=BC_MODEL,
        kind="baseline",
    ))
    if len(panels) != GRID_COLUMNS * GRID_ROWS:
        raise AssertionError(f"expected 25 panels, got {len(panels)}")
    missing = [str(panel.model_path) for panel in panels if not panel.model_path.is_file()]
    if missing:
        raise FileNotFoundError("missing policy checkpoint(s):\n" + "\n".join(missing))
    return panels


def render_policy_frames(
    env: ProneKeyframeTransitionEnv,
    renderer: mujoco.Renderer,
    camera: mujoco.MjvCamera,
    panel: PolicyPanel,
) -> list[bytes]:
    policy = PPO.load(panel.model_path, device="cpu")
    observation, _ = env.reset(seed=0)
    ended = False
    encoded_frames: list[bytes] = []
    for _ in range(FRAMES):
        for _ in range(2):
            if ended:
                break
            action, _ = policy.predict(observation, deterministic=True)
            observation, _, terminated, truncated, _ = env.step(action)
            ended = terminated or truncated
        renderer.update_scene(env.data, camera=camera)
        image = Image.fromarray(renderer.render())
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=78, optimize=True)
        encoded_frames.append(buffer.getvalue())
    del policy
    return encoded_frames


def compose_frame(
    panels: list[PolicyPanel],
    tile_frames: list[list[bytes]],
    frame_number: int,
    bold_font: ImageFont.ImageFont,
    small_font: ImageFont.ImageFont,
) -> np.ndarray:
    width = GRID_COLUMNS * PANEL_WIDTH
    height = GRID_ROWS * (HEADER_HEIGHT + PANEL_HEIGHT) + FOOTER_HEIGHT
    canvas = Image.new("RGB", (width, height), (17, 22, 29))
    draw = ImageDraw.Draw(canvas)
    header_colors = {
        "success": (34, 91, 68),
        "failure": (104, 48, 50),
        "baseline": (119, 89, 28),
    }
    for index, panel in enumerate(panels):
        row, column = divmod(index, GRID_COLUMNS)
        x = column * PANEL_WIDTH
        y = row * (HEADER_HEIGHT + PANEL_HEIGHT)
        draw.rectangle((x, y, x + PANEL_WIDTH - 1, y + HEADER_HEIGHT - 1),
                       fill=header_colors[panel.kind])
        draw.text((x + 8, y + 3), panel.title, fill=(255, 255, 255), font=bold_font)
        draw.text((x + 8, y + 21), panel.subtitle,
                  fill=(222, 228, 235), font=small_font)
        tile = Image.open(io.BytesIO(tile_frames[index][frame_number])).convert("RGB")
        canvas.paste(tile, (x, y + HEADER_HEIGHT))
        draw.rectangle((x, y, x + PANEL_WIDTH - 1,
                        y + HEADER_HEIGHT + PANEL_HEIGHT - 1),
                       outline=(76, 84, 96), width=1)

    footer_y = height - FOOTER_HEIGHT
    draw.text((12, footer_y + 4),
              "Same prone start and camera  |  6 simulated seconds  |  "
              "PASS = both feet planted, torso tip >= 0.50 m, foot X speed <= 0.02 m/s for 2 s",
              fill=(233, 237, 242), font=small_font)
    draw.text((12, footer_y + 23),
              f"8-100 seed sweep: 11/93 policies passed 20/20 evaluations  |  "
              f"frame {frame_number + 1}/{FRAMES}",
              fill=(171, 183, 198), font=small_font)
    return np.asarray(canvas)


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    panels = panels_from_summary()
    env = ProneKeyframeTransitionEnv(reward_mode="height_sway")
    renderer = mujoco.Renderer(env.model, height=PANEL_HEIGHT, width=PANEL_WIDTH)
    camera = camera_for_scene()

    tile_frames: list[list[bytes]] = []
    try:
        for index, panel in enumerate(panels, start=1):
            print(f"Rendering panel {index}/25: {panel.title}", flush=True)
            tile_frames.append(render_policy_frames(env, renderer, camera, panel))
    finally:
        renderer.close()
        env.close()

    ffmpeg = shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"
    output_width = GRID_COLUMNS * PANEL_WIDTH
    output_height = GRID_ROWS * (HEADER_HEIGHT + PANEL_HEIGHT) + FOOTER_HEIGHT
    process = subprocess.Popen(
        [
            ffmpeg,
            "-y",
            "-loglevel", "error",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-s", f"{output_width}x{output_height}",
            "-pix_fmt", "rgb24",
            "-r", str(FPS),
            "-i", "-",
            "-an",
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "20",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            str(OUTPUT),
        ],
        stdin=subprocess.PIPE,
    )
    assert process.stdin is not None
    bold_font = load_font(14, bold=True)
    small_font = load_font(11)
    poster_frame: np.ndarray | None = None
    try:
        for frame_number in range(FRAMES):
            frame = compose_frame(panels, tile_frames, frame_number,
                                  bold_font, small_font)
            if frame_number == POSTER_FRAME:
                poster_frame = frame
            process.stdin.write(frame.tobytes())
    finally:
        process.stdin.close()
    return_code = process.wait()
    if return_code:
        raise RuntimeError(f"ffmpeg exited with status {return_code}")
    assert poster_frame is not None
    Image.fromarray(poster_frame).save(POSTER, quality=88, optimize=True)
    print(f"Wrote {OUTPUT}")
    print(f"Wrote {POSTER}")


if __name__ == "__main__":
    main()
