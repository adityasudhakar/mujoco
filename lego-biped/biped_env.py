"""Gymnasium environment for learning the planar LEGO biped get-up motion."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np


MODEL_PATH = Path(__file__).with_name("lego_biped.xml")
CONTROL_HZ = 50
CONTROL_DT = 1.0 / CONTROL_HZ
EPISODE_SECONDS = 8.0
SUCCESS_HOLD_SECONDS = 2.0


class LegoBipedGetupEnv(gym.Env[np.ndarray, np.ndarray]):
    """Learn joint-position increments that bring the biped up from prone."""

    metadata = {"render_modes": [], "render_fps": CONTROL_HZ}

    joint_names = ("left_hip", "left_knee", "right_hip", "right_knee")
    actuator_names = (
        "left_hip_servo",
        "left_knee_servo",
        "right_hip_servo",
        "right_knee_servo",
    )

    def __init__(
        self,
        *,
        reset_mode: str = "train",
        max_episode_steps: int = round(EPISODE_SECONDS * CONTROL_HZ),
    ) -> None:
        super().__init__()
        if reset_mode not in {"train", "prone", "stand"}:
            raise ValueError(f"unknown reset_mode {reset_mode!r}")

        self.model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
        self.data = mujoco.MjData(self.model)
        self.reset_mode = reset_mode
        self.max_episode_steps = max_episode_steps
        self.physics_steps_per_action = round(CONTROL_DT / self.model.opt.timestep)
        if not np.isclose(
            self.physics_steps_per_action * self.model.opt.timestep, CONTROL_DT
        ):
            raise ValueError("model timestep must divide the 50 Hz control period")

        self.joint_ids = np.asarray(
            [self._id(mujoco.mjtObj.mjOBJ_JOINT, name) for name in self.joint_names],
            dtype=np.int32,
        )
        self.qpos_adrs = self.model.jnt_qposadr[self.joint_ids]
        self.dof_adrs = self.model.jnt_dofadr[self.joint_ids]
        self.actuator_ids = np.asarray(
            [
                self._id(mujoco.mjtObj.mjOBJ_ACTUATOR, name)
                for name in self.actuator_names
            ],
            dtype=np.int32,
        )

        self.root_x_adr = self.model.jnt_qposadr[
            self._id(mujoco.mjtObj.mjOBJ_JOINT, "root_x")
        ]
        self.root_z_adr = self.model.jnt_qposadr[
            self._id(mujoco.mjtObj.mjOBJ_JOINT, "root_z")
        ]
        self.root_pitch_joint = self._id(mujoco.mjtObj.mjOBJ_JOINT, "root_pitch")
        self.root_pitch_adr = self.model.jnt_qposadr[self.root_pitch_joint]
        self.root_dof_adrs = np.asarray(
            [
                self.model.jnt_dofadr[
                    self._id(mujoco.mjtObj.mjOBJ_JOINT, joint_name)
                ]
                for joint_name in ("root_x", "root_z", "root_pitch")
            ],
            dtype=np.int32,
        )

        self.torso_body = self._id(mujoco.mjtObj.mjOBJ_BODY, "torso")
        self.foot_bodies = np.asarray(
            [
                self._id(mujoco.mjtObj.mjOBJ_BODY, "left_foot"),
                self._id(mujoco.mjtObj.mjOBJ_BODY, "right_foot"),
            ],
            dtype=np.int32,
        )
        self.ground_geom = self._id(mujoco.mjtObj.mjOBJ_GEOM, "ground")
        self.torso_geom = self._id(mujoco.mjtObj.mjOBJ_GEOM, "torso_geom")
        self.foot_geoms = np.asarray(
            [
                self._id(mujoco.mjtObj.mjOBJ_GEOM, "left_foot_geom"),
                self._id(mujoco.mjtObj.mjOBJ_GEOM, "right_foot_geom"),
            ],
            dtype=np.int32,
        )

        self.stand_key = self._id(mujoco.mjtObj.mjOBJ_KEY, "stand")
        self.prone_key = self._id(mujoco.mjtObj.mjOBJ_KEY, "prone")
        self.stand_qpos = self.model.key_qpos[self.stand_key].copy()
        self.prone_qpos = self.model.key_qpos[self.prone_key].copy()

        joint_ranges = self.model.jnt_range[self.joint_ids]
        actuator_ranges = self.model.actuator_ctrlrange[self.actuator_ids]
        self.target_low = np.maximum(joint_ranges[:, 0], actuator_ranges[:, 0])
        self.target_high = np.minimum(joint_ranges[:, 1], actuator_ranges[:, 1])
        # At full action, move a target by 6 degrees per 20 ms (300 deg/s).
        self.max_target_delta = np.deg2rad(6.0)

        self.action_space = spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)
        self.observation_space = spaces.Box(
            -10.0, 10.0, shape=(25,), dtype=np.float32
        )

        self.robot_weight = float(self.model.body_mass.sum() * abs(self.model.opt.gravity[2]))
        self.curriculum_difficulty = 0.15
        self.targets = np.zeros(4, dtype=np.float64)
        self.step_count = 0
        self.standing_steps = 0
        self.previous_potential = 0.0
        self.previous_foot_x = np.zeros(2, dtype=np.float64)
        self._contact_wrench = np.zeros(6, dtype=np.float64)

    def _id(self, obj_type: mujoco.mjtObj, name: str) -> int:
        obj_id = mujoco.mj_name2id(self.model, obj_type, name)
        if obj_id < 0:
            raise ValueError(f"model is missing {name!r}")
        return obj_id

    def set_curriculum_difficulty(self, difficulty: float) -> None:
        """Set reset difficulty from 0 (near standing) to 1 (fully prone)."""
        self.curriculum_difficulty = float(np.clip(difficulty, 0.0, 1.0))

    def _training_reset_fraction(self) -> float:
        difficulty = self.curriculum_difficulty
        # Follow a moving window toward prone instead of retaining already-
        # standing resets. At full difficulty, every reset is at least 70% of
        # the way to prone and 45% use the exact acceptance-test pose.
        if self.np_random.random() < 0.45:
            return difficulty
        lower = max(0.0, difficulty - 0.30)
        return float(self.np_random.uniform(lower, difficulty))

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        if options is not None and "reset_fraction" in options:
            reset_fraction = float(np.clip(options["reset_fraction"], 0.0, 1.0))
        elif self.reset_mode == "prone":
            reset_fraction = 1.0
        elif self.reset_mode == "stand":
            reset_fraction = 0.0
        else:
            reset_fraction = self._training_reset_fraction()

        self.data.qpos[:] = (
            (1.0 - reset_fraction) * self.stand_qpos
            + reset_fraction * self.prone_qpos
        )

        if self.reset_mode == "train":
            noise_scale = 0.02 + 0.02 * self.curriculum_difficulty
            self.data.qpos[self.root_x_adr] += self.np_random.uniform(-0.015, 0.015)
            self.data.qpos[self.root_pitch_adr] += self.np_random.normal(
                0.0, noise_scale
            )
            self.data.qpos[self.qpos_adrs] += self.np_random.normal(
                0.0, noise_scale, size=4
            )
            self.data.qvel[:] = self.np_random.normal(0.0, 0.015, self.model.nv)

        self.targets[:] = np.clip(
            self.data.qpos[self.qpos_adrs], self.target_low, self.target_high
        )
        self.data.ctrl[self.actuator_ids] = self.targets
        mujoco.mj_forward(self.model, self.data)

        self.step_count = 0
        self.standing_steps = 0
        self.previous_foot_x[:] = self.data.xpos[self.foot_bodies, 0]
        metrics = self._metrics()
        self.previous_potential = self._potential(metrics)

        info = {
            **metrics,
            "reset_fraction": reset_fraction,
            "curriculum_difficulty": self.curriculum_difficulty,
            "is_success": False,
        }
        return self._observation(metrics), info

    def _contact_forces(self) -> tuple[np.ndarray, float]:
        foot_forces = np.zeros(2, dtype=np.float64)
        torso_force = 0.0
        for contact_index in range(self.data.ncon):
            contact = self.data.contact[contact_index]
            if contact.efc_address < 0:
                continue
            geom1, geom2 = int(contact.geom[0]), int(contact.geom[1])
            if self.ground_geom not in (geom1, geom2):
                continue
            other_geom = geom2 if geom1 == self.ground_geom else geom1
            self._contact_wrench.fill(0.0)
            mujoco.mj_contactForce(
                self.model, self.data, contact_index, self._contact_wrench
            )
            normal_force = max(0.0, float(self._contact_wrench[0]))
            if other_geom == self.torso_geom:
                torso_force += normal_force
            else:
                matches = np.flatnonzero(self.foot_geoms == other_geom)
                if matches.size:
                    foot_forces[int(matches[0])] += normal_force
        return foot_forces, torso_force

    def _metrics(self) -> dict[str, Any]:
        torso_x = float(self.data.xpos[self.torso_body, 0])
        torso_z = float(self.data.xpos[self.torso_body, 2])
        upright = float(self.data.xmat[self.torso_body].reshape(3, 3)[2, 2])
        foot_forces, torso_force = self._contact_forces()
        return {
            "torso_x": torso_x,
            "torso_z": torso_z,
            "upright": upright,
            "left_foot_force": float(foot_forces[0]),
            "right_foot_force": float(foot_forces[1]),
            "torso_ground_force": torso_force,
        }

    @staticmethod
    def _potential(metrics: dict[str, Any]) -> float:
        height = float(np.clip((metrics["torso_z"] - 0.035) / 0.285, 0.0, 1.0))
        upright = float(np.clip((metrics["upright"] + 0.05) / 1.05, 0.0, 1.0))
        # Upright-but-sitting was a strong local optimum in the first run, so
        # uprightness is valuable mainly when the torso is also elevated.
        return 3.0 * height + 1.5 * height * upright + 0.25 * upright

    def _observation(self, metrics: dict[str, Any]) -> np.ndarray:
        pitch = float(self.data.qpos[self.root_pitch_adr])
        root_vel = self.data.qvel[self.root_dof_adrs]
        joint_pos = self.data.qpos[self.qpos_adrs]
        joint_vel = self.data.qvel[self.dof_adrs]
        foot_relative = self.data.xpos[self.foot_bodies][:, (0, 2)] - self.data.xpos[
            self.torso_body, (0, 2)
        ]
        target_mid = 0.5 * (self.target_low + self.target_high)
        target_half_range = 0.5 * (self.target_high - self.target_low)
        normalized_targets = (self.targets - target_mid) / target_half_range

        obs = np.concatenate(
            [
                np.asarray(
                    [
                        metrics["torso_x"],
                        metrics["torso_z"] / 0.4,
                        np.sin(pitch),
                        np.cos(pitch),
                    ]
                ),
                root_vel / np.asarray([3.0, 3.0, 10.0]),
                joint_pos / np.pi,
                joint_vel / 20.0,
                foot_relative.reshape(-1) / 0.4,
                np.asarray(
                    [metrics["left_foot_force"], metrics["right_foot_force"]]
                )
                / max(self.robot_weight, 1e-6),
                normalized_targets,
            ]
        )
        return np.clip(obs, -10.0, 10.0).astype(np.float32)

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        if action.shape != (4,):
            raise ValueError(f"expected action shape (4,), got {action.shape}")

        self.targets[:] = np.clip(
            self.targets + action * self.max_target_delta,
            self.target_low,
            self.target_high,
        )
        self.data.ctrl[self.actuator_ids] = self.targets
        for _ in range(self.physics_steps_per_action):
            mujoco.mj_step(self.model, self.data)

        self.step_count += 1
        metrics = self._metrics()
        potential = self._potential(metrics)
        potential_progress = potential - self.previous_potential
        self.previous_potential = potential

        height = float(np.clip((metrics["torso_z"] - 0.035) / 0.285, 0.0, 1.0))
        upright = float(np.clip((metrics["upright"] + 0.05) / 1.05, 0.0, 1.0))
        standing = metrics["torso_z"] > 0.28 and metrics["upright"] > 0.90
        self.standing_steps = self.standing_steps + 1 if standing else 0
        success = self.standing_steps >= round(SUCCESS_HOLD_SECONDS * CONTROL_HZ)

        current_foot_x = self.data.xpos[self.foot_bodies, 0].copy()
        foot_force_array = np.asarray(
            [metrics["left_foot_force"], metrics["right_foot_force"]]
        )
        planted = foot_force_array > 0.2
        foot_speed = np.abs(current_foot_x - self.previous_foot_x) / CONTROL_DT
        foot_slip = float(np.sum(foot_speed[planted]))
        self.previous_foot_x[:] = current_foot_x
        torso_load = float(
            np.clip(metrics["torso_ground_force"] / self.robot_weight, 0.0, 1.5)
        )

        reward = (
            1.5 * height
            + 2.5 * height * upright
            + 10.0 * potential_progress
            + 3.0 * float(standing)
            + 75.0 * float(success)
            - 0.002 * float(np.square(action).sum())
            - 0.01 * foot_slip
            - 1.25 * torso_load
        )

        finite = bool(
            np.all(np.isfinite(self.data.qpos)) and np.all(np.isfinite(self.data.qvel))
        )
        out_of_bounds = abs(metrics["torso_x"]) > 1.5 or metrics["torso_z"] < -0.1
        terminated = bool(success or not finite or out_of_bounds)
        truncated = bool(self.step_count >= self.max_episode_steps and not terminated)
        if not finite:
            reward = -100.0

        info = {
            **metrics,
            "height_score": height,
            "upright_score": upright,
            "foot_slip": foot_slip,
            "torso_load": torso_load,
            "standing_steps": self.standing_steps,
            "is_success": success,
            "curriculum_difficulty": self.curriculum_difficulty,
        }
        return self._observation(metrics), float(reward), terminated, truncated, info


if __name__ == "__main__":
    from stable_baselines3.common.env_checker import check_env

    environment = LegoBipedGetupEnv()
    check_env(environment, warn=True)
    observation, _ = environment.reset(seed=0)
    total_reward = 0.0
    for _ in range(1_000):
        observation, reward, terminated, truncated, _ = environment.step(
            environment.action_space.sample()
        )
        total_reward += reward
        if terminated or truncated:
            observation, _ = environment.reset()
    print(f"environment check passed; random-rollout reward={total_reward:.1f}")
