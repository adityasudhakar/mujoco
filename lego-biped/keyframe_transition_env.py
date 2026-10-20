"""RL task for getting up from prone and standing with minimal foot sway."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import mujoco
import numpy as np

from biped_env import CONTROL_DT, SUCCESS_HOLD_SECONDS, LegoBipedGetupEnv
from scripted_motions import MOTIONS_DEG


MIN_FOOT_CONTACT_FORCE = 0.2
MIN_SUCCESS_TORSO_TOP_Z = 0.50
MAX_SUCCESS_FOOT_X_SPEED = 0.02


class ProneKeyframeTransitionEnv(LegoBipedGetupEnv):
    """Start at the prone keyframe and learn a low-sway full get-up."""

    metadata = {"render_modes": [], "render_fps": 50}

    def __init__(
        self, max_episode_steps: int = 300, reward_mode: str = "original"
    ) -> None:
        super().__init__(reset_mode="prone", max_episode_steps=max_episode_steps)
        if reward_mode not in {"original", "height_sway"}:
            raise ValueError(f"unknown reward_mode {reward_mode!r}")
        self.reward_mode = reward_mode
        # The first scripted pose intentionally exceeds the hip's joint
        # limit; MuJoCo's actuator saturates it while the joint limit
        # constrains motion.
        self.target_low = np.minimum(self.target_low, np.deg2rad([-180, -150, -180, -150]))
        # The first 0.5 s scripted segment changes the hip target by up to
        # about 11 degrees per 20 ms; allow the policy to reproduce it.
        self.max_target_delta = np.deg2rad(12.0)
        self.prone_qpos, self.prone_qvel, self.prone_targets = self._prone_pose()
        final_segment = len(MOTIONS_DEG["getup_prone_1"]) - 1
        self.good_qpos, _, self.good_targets = self._reference_pose(final_segment)
        self.good_joint_qpos = self.good_qpos[self.qpos_adrs].copy()
        self.good_pitch = float(self.good_qpos[self.root_pitch_adr])
        self.root_z_adr = self.model.jnt_qposadr[
            self._id(mujoco.mjtObj.mjOBJ_JOINT, "root_z")
        ]
        self.good_root_z = float(self.good_qpos[self.root_z_adr])
        self.previous_goal_distance = 0.0
        self.previous_action = np.zeros(4, dtype=np.float64)

    def _prone_pose(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Capture the exact prone keyframe, before any get-up motion runs."""
        data = mujoco.MjData(self.model)
        prone_id = self._id(mujoco.mjtObj.mjOBJ_KEY, "prone")
        mujoco.mj_resetDataKeyframe(self.model, data, prone_id)
        return (
            data.qpos.copy(),
            data.qvel.copy(),
            data.ctrl[self.actuator_ids].copy(),
        )

    def _reference_pose(self, through_segment: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Replay the scripted motion exactly and capture a segment endpoint."""
        model = self.model
        data = mujoco.MjData(model)
        prone_id = self._id(mujoco.mjtObj.mjOBJ_KEY, "prone")
        mujoco.mj_resetDataKeyframe(model, data, prone_id)
        start_targets = data.ctrl[self.actuator_ids].copy()
        physics_steps = round(CONTROL_DT / model.opt.timestep)

        for segment_index, (duration, target_degrees) in enumerate(
            MOTIONS_DEG["getup_prone_1"][: through_segment + 1]
        ):
            target = np.deg2rad(np.asarray(target_degrees, dtype=np.float64))
            ticks = max(1, round(duration * 50))
            segment_start = start_targets.copy()
            for tick in range(ticks):
                blend = (tick + 1) / ticks
                blend = blend * blend * (3.0 - 2.0 * blend)
                data.ctrl[self.actuator_ids] = segment_start + blend * (
                    target - segment_start
                )
                for _ in range(physics_steps):
                    mujoco.mj_step(model, data)
            start_targets = target
            if segment_index == through_segment:
                return data.qpos.copy(), data.qvel.copy(), target.copy()

        raise RuntimeError("reference motion did not contain the requested segment")

    def _goal_distance(self) -> float:
        joint_error = (self.data.qpos[self.qpos_adrs] - self.good_joint_qpos) / np.deg2rad(20)
        pitch_error = (self.data.qpos[self.root_pitch_adr] - self.good_pitch) / np.deg2rad(15)
        height_error = (self.data.qpos[self.root_z_adr] - self.good_root_z) / 0.04
        return float(np.sqrt(np.mean(np.square(np.concatenate(
            [joint_error, np.asarray([pitch_error, height_error])]
        )))))

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self.data.qpos[:] = self.prone_qpos
        self.data.qvel[:] = self.prone_qvel
        self.targets[:] = np.clip(
            self.prone_targets, self.target_low, self.target_high
        )
        self.data.ctrl[self.actuator_ids] = self.targets
        mujoco.mj_forward(self.model, self.data)
        self.step_count = 0
        self.standing_steps = 0
        self.previous_action[:] = 0.0
        self.previous_foot_x[:] = self.data.xpos[self.foot_bodies, 0]
        metrics = self._metrics()
        self.previous_goal_distance = self._goal_distance()
        info = {**metrics, "goal_distance": self.previous_goal_distance, "is_success": False}
        return self._observation(metrics), info

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
        foot_forces = np.asarray(
            [metrics["left_foot_force"], metrics["right_foot_force"]]
        )
        planted = foot_forces > MIN_FOOT_CONTACT_FORCE
        planted_count = int(np.count_nonzero(planted))
        current_foot_x = self.data.xpos[self.foot_bodies, 0].copy()
        foot_speed = np.abs(current_foot_x - self.previous_foot_x) / CONTROL_DT
        foot_sway = float(np.sum(foot_speed[planted]))
        self.previous_foot_x[:] = current_foot_x

        torso_rotation = self.data.xmat[self.torso_body].reshape(3, 3)
        torso_top_z = float(
            self.data.xpos[self.torso_body, 2]
            + (torso_rotation @ np.asarray([0.0, 0.0, 0.216]))[2]
        )

        goal_distance = self._goal_distance()
        progress = self.previous_goal_distance - goal_distance
        self.previous_goal_distance = goal_distance
        standing = metrics["torso_z"] > 0.28 and metrics["upright"] > 0.90
        both_feet_planted = planted_count == 2
        max_foot_x_speed = float(np.max(foot_speed))
        goal_pose = (
            torso_top_z >= MIN_SUCCESS_TORSO_TOP_Z
            and both_feet_planted
            and max_foot_x_speed <= MAX_SUCCESS_FOOT_X_SPEED
        )
        self.standing_steps = self.standing_steps + 1 if goal_pose else 0
        success = self.standing_steps >= round(SUCCESS_HOLD_SECONDS / CONTROL_DT)

        height_score = float(np.clip((metrics["torso_z"] - 0.035) / 0.285, 0.0, 1.0))
        upright_score = float(np.clip((metrics["upright"] + 0.05) / 1.05, 0.0, 1.0))
        if self.reward_mode == "height_sway":
            # Shape all three goals: raise the torso tip, reward both planted
            # feet, and penalize horizontal movement of planted feet. Success
            # additionally requires all three together for the full hold.
            foot_plant_reward = 2.0 * planted_count
            reward = (
                10.0 * torso_top_z
                - 5.0 * foot_sway
                + foot_plant_reward
            )
        else:
            reward = (
                20.0 * progress
                - 0.08 * goal_distance
                + 1.0 * height_score
                + 1.0 * height_score * upright_score
                + 3.0 * float(standing)
                + 75.0 * float(success)
                # A planted foot should stay put throughout the get-up and the
                # two-second stand hold.  The previous 0.01 coefficient was
                # effectively negligible beside the height/standing rewards.
                - 5.0 * foot_sway
                - 0.005 * float(np.square(action - self.previous_action).sum())
                - 0.01 * float(np.square(action).sum())
            )
        self.previous_action[:] = action

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
            "goal_distance": goal_distance,
            "foot_sway": foot_sway,
            "torso_top_z": torso_top_z,
            "planted_count": planted_count,
            "max_foot_x_speed": max_foot_x_speed,
            "foot_plant_reward": (
                foot_plant_reward if self.reward_mode == "height_sway" else 0.0
            ),
            "is_success": success,
            "standing_steps": self.standing_steps,
            "target_deg": np.rad2deg(self.targets).copy(),
        }
        return self._observation(metrics), float(reward), terminated, truncated, info
