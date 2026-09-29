"""Watch the trained Gymnasium InvertedPendulum policy."""
import gymnasium as gym
from stable_baselines3 import PPO
import time

print("Loading trained policy...")
model = PPO.load("checkpoints/inverted_pendulum_ppo")

print("Starting visualization (close window to exit)")
env = gym.make("InvertedPendulum-v5", render_mode="human")

for episode in range(10):
    obs, _ = env.reset()
    total_reward = 0

    for step in range(1000):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        time.sleep(0.002)  # slow down for visibility

        if terminated or truncated:
            print(f"Episode {episode+1}: Fell at step {step}")
            break
    else:
        print(f"Episode {episode+1}: Survived all 1000 steps!")

env.close()
