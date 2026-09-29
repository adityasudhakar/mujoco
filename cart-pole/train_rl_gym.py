"""Train RL on Gymnasium's InvertedPendulum - a known-working cart-pole.

This uses the pre-made environment to verify RL works, then we can
debug why our custom environment doesn't.
"""
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import time

def train():
    print("=" * 50)
    print("Training on Gymnasium InvertedPendulum-v5")
    print("(This is a known-working MuJoCo cart-pole)")
    print("=" * 50)

    # Create environment - this is MuJoCo's cart-pole
    env = make_vec_env("InvertedPendulum-v5", n_envs=4)

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        device="cpu",
    )

    print("\nTraining for 500,000 timesteps...")
    start = time.time()
    model.learn(total_timesteps=500000)
    elapsed = time.time() - start

    model.save("checkpoints/inverted_pendulum_ppo")
    print(f"\nTraining complete in {elapsed:.1f}s")
    print("Saved to checkpoints/inverted_pendulum_ppo.zip")

    # Test it
    print("\n" + "=" * 50)
    print("Testing trained policy (10 episodes)")
    print("=" * 50)

    test_env = gym.make("InvertedPendulum-v5")

    successes = 0
    for ep in range(10):
        obs, _ = test_env.reset()
        total_reward = 0
        for step in range(1000):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = test_env.step(action)
            total_reward += reward
            if terminated or truncated:
                break

        if step >= 999:
            successes += 1
            print(f"  Episode {ep+1}: SURVIVED 1000 steps (reward={total_reward:.0f})")
        else:
            print(f"  Episode {ep+1}: Fell at step {step} (reward={total_reward:.0f})")

    print(f"\nSuccess rate: {successes}/10 episodes survived full length")

    env.close()
    test_env.close()

if __name__ == "__main__":
    train()
