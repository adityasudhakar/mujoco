#!/usr/bin/env python3
"""Train the prone get-up with torso-tip height and planted-foot sway rewards."""

from __future__ import annotations

import argparse
from functools import partial
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor

from keyframe_transition_env import ProneKeyframeTransitionEnv
from train_keyframe_transition import (
    behavior_clone,
    collect_script_demonstration,
    evaluate_policy,
)


RUNS_DIR = Path(__file__).with_name("checkpoints")


def make_env() -> ProneKeyframeTransitionEnv:
    return ProneKeyframeTransitionEnv(reward_mode="height_sway")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=300_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-name", type=str, default=None)
    args = parser.parse_args()

    run_name = args.run_name or f"prone_height_sway_balanced_10_seed{args.seed}"
    run_dir = RUNS_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    training_env = make_vec_env(
        make_env, n_envs=args.n_envs, seed=args.seed
    )
    evaluation_env = Monitor(make_env())
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
    # Demonstration collection uses the same start state and controller limits;
    # only its reward differs, so this keeps the warm start comparable.
    observations, actions, reached_goal = collect_script_demonstration()
    print(f"Scripted demonstration reached good keyframe: {reached_goal}")
    behavior_clone(model, observations, actions)
    bc_successes, bc_sway, bc_reward = evaluate_policy(model, evaluation_env, 10)
    print(
        f"Behavior-cloning baseline: {bc_successes}/10 full stands; "
        f"mean planted-foot sway={bc_sway:.5f} m/s; mean reward={bc_reward:.2f}"
    )
    best_dir = run_dir / "best"
    best_dir.mkdir(parents=True, exist_ok=True)
    if bc_successes == 10:
        model.save(best_dir / "best_model")

    evaluation_callback = EvalCallback(
        evaluation_env,
        log_path=str(run_dir / "eval"),
        eval_freq=max(10_000 // args.n_envs, 1),
        n_eval_episodes=10,
        deterministic=True,
    )
    evaluation_callback.best_mean_reward = bc_reward
    callbacks = [
        CheckpointCallback(
            save_freq=max(50_000 // args.n_envs, 1),
            save_path=str(run_dir),
            name_prefix="height_sway_ppo",
        ),
        evaluation_callback,
    ]
    model.learn(total_timesteps=args.timesteps, callback=callbacks)
    model.save(run_dir / "height_sway_ppo_final")
    final_policy = PPO.load(run_dir / "height_sway_ppo_final.zip")
    successes, mean_sway, mean_reward = evaluate_policy(
        final_policy, evaluation_env, 20
    )
    print(
        f"Final deterministic policy: {successes}/20 full stands; "
        f"mean planted-foot sway={mean_sway:.5f} m/s; "
        f"mean reward={mean_reward:.2f}"
    )
    training_env.close()
    evaluation_env.close()


if __name__ == "__main__":
    main()
