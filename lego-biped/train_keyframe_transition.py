#!/usr/bin/env python3
"""Train PPO to get up from prone while minimizing planted-foot sway."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor

from keyframe_transition_env import ProneKeyframeTransitionEnv
from scripted_motions import MOTIONS_DEG


RUN_DIR = Path(__file__).with_name("checkpoints") / "prone_keyframe_transition"


def collect_script_demonstration() -> tuple[np.ndarray, np.ndarray, bool]:
    """Record the known good transition as a warm start for PPO's actor."""
    env = ProneKeyframeTransitionEnv()
    observation, _ = env.reset(seed=0)
    observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    for duration, target_degrees in MOTIONS_DEG["getup_prone_1"]:
        segment_start = env.targets.copy()
        target = np.deg2rad(np.asarray(target_degrees, dtype=np.float64))
        ticks = round(duration * 50)
        for tick in range(ticks):
            blend = (tick + 1) / ticks
            blend = blend * blend * (3.0 - 2.0 * blend)
            next_targets = segment_start + blend * (target - segment_start)
            action = (next_targets - env.targets) / env.max_target_delta
            action = np.clip(action, -1.0, 1.0).astype(np.float32)
            observations.append(observation.copy())
            actions.append(action)
            observation, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        if terminated or truncated:
            break
    for _ in range(round(2.0 * 50)):
        action = np.zeros(4, dtype=np.float32)
        observations.append(observation.copy())
        actions.append(action)
        observation, _, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    env.close()
    return np.asarray(observations), np.asarray(actions), bool(info["is_success"])


def behavior_clone(model: PPO, observations: np.ndarray, actions: np.ndarray) -> None:
    """Initialize PPO's actor by imitating the scripted successful transition."""
    obs_tensor = torch.as_tensor(observations, device=model.device)
    action_tensor = torch.as_tensor(actions, device=model.device)
    model.policy.set_training_mode(True)
    # Full prone-to-stand demonstrations need a tighter actor fit than the
    # shorter bad-pose transition used in the earlier experiment.
    for _ in range(5_000):
        distribution = model.policy.get_distribution(obs_tensor)
        predicted = distribution.distribution.mean
        loss = torch.nn.functional.mse_loss(predicted, action_tensor)
        model.policy.optimizer.zero_grad()
        loss.backward()
        model.policy.optimizer.step()
    print(f"Behavior-cloned actor from {len(observations)} scripted transitions")


def evaluate_policy(
    model: PPO, env: ProneKeyframeTransitionEnv, episodes: int
) -> tuple[int, float, float]:
    successes = 0
    sway_sums: list[float] = []
    episode_returns: list[float] = []
    for episode in range(episodes):
        observation, _ = env.reset(seed=episode)
        episode_sway = 0.0
        episode_return = 0.0
        episode_steps = 0
        while True:
            action, _ = model.predict(observation, deterministic=True)
            observation, reward, terminated, truncated, info = env.step(action)
            episode_return += reward
            episode_sway += info["foot_sway"]
            episode_steps += 1
            if terminated or truncated:
                successes += int(info["is_success"])
                sway_sums.append(episode_sway / max(1, episode_steps))
                episode_returns.append(episode_return)
                break
    return (
        successes,
        float(np.mean(sway_sums)),
        float(np.mean(episode_returns)),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=300_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bc-only", action="store_true")
    args = parser.parse_args()

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    training_env = make_vec_env(
        ProneKeyframeTransitionEnv, n_envs=args.n_envs, seed=args.seed
    )
    evaluation_env = Monitor(ProneKeyframeTransitionEnv())
    model = PPO(
        "MlpPolicy",
        training_env,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        policy_kwargs={"net_arch": {"pi": [128, 128], "vf": [128, 128]}},
        verbose=1,
        seed=args.seed,
        device="cpu",
    )
    observations, actions, demonstration_reached_goal = collect_script_demonstration()
    print(f"Scripted demonstration reached good keyframe: {demonstration_reached_goal}")
    behavior_clone(model, observations, actions)
    bc_successes, bc_sway, bc_mean_reward = evaluate_policy(
        model, evaluation_env, 10
    )
    print(
        f"Behavior-cloning baseline: {bc_successes}/10 full stands; "
        f"mean planted-foot sway={bc_sway:.5f} m/s; "
        f"mean reward={bc_mean_reward:.2f}"
    )
    best_model_path = RUN_DIR / "best" / "best_model"
    best_model_path.parent.mkdir(parents=True, exist_ok=True)
    if bc_successes == 10:
        model.save(best_model_path)
    if args.bc_only:
        model.save(RUN_DIR / "transition_ppo_bc_only")
        training_env.close()
        evaluation_env.close()
        return
    evaluation_callback = EvalCallback(
        evaluation_env,
        log_path=str(RUN_DIR / "eval"),
        eval_freq=max(10_000 // args.n_envs, 1),
        n_eval_episodes=10,
        deterministic=True,
    )
    # PPO checkpoints only replace the successful behavior-cloned baseline
    # when their evaluation return, including the sway penalty, is better.
    evaluation_callback.best_mean_reward = bc_mean_reward
    callbacks = [
        CheckpointCallback(
            save_freq=max(50_000 // args.n_envs, 1),
            save_path=str(RUN_DIR),
            name_prefix="transition_ppo",
        ),
        evaluation_callback,
    ]
    model.learn(total_timesteps=args.timesteps, callback=callbacks)
    model.save(RUN_DIR / "transition_ppo_final")

    final_policy = PPO.load(RUN_DIR / "transition_ppo_final.zip")
    successes, mean_sway, mean_reward = evaluate_policy(final_policy, evaluation_env, 20)
    print(
        f"Final deterministic policy: {successes}/20 full stands; "
        f"mean planted-foot sway={mean_sway:.5f} m/s; "
        f"mean reward={mean_reward:.2f}"
    )
    training_env.close()
    evaluation_env.close()


if __name__ == "__main__":
    main()
