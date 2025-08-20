import os
import time
import pandas as pd
import numpy as np
import minihack_env as me
import commons
from gymnasium import ObservationWrapper, spaces
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.monitor import Monitor


class MiniHackWrapper(ObservationWrapper):
    def __init__(self, env, chars_size=(5, 5)):
        super().__init__(env)
        self.chars_size = chars_size
        self.observation_space = spaces.Box(low=0, high=255, shape=chars_size, dtype=np.uint8)

    def observation(self, obs):
        chars = commons.get_crop_chars_from_observation(obs)
        return chars[:self.chars_size[0], :self.chars_size[1]].astype(np.uint8)

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        return self.observation(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self.observation(obs), reward, terminated, truncated, info


def train_agent(agent_type, env_id, chars_size, total_timesteps=2_000_000):
    env_name = env_id.split(".")[-1]
    agent_name = agent_type.__name__.lower()
    log_dir = f"./results_dl/{agent_name}_{env_name}_{chars_size[0]}x{chars_size[1]}"
    os.makedirs(log_dir, exist_ok=True)

    print(f"\nTraining {agent_type.__name__} on {env_name} with chars_size={chars_size}")

    raw_env = me.get_minihack_envirnment(env_id)
    wrapped_env = Monitor(MiniHackWrapper(raw_env, chars_size=chars_size))

    model = agent_type("MlpPolicy", wrapped_env, verbose=1)

    start_time = time.time()
    model.learn(total_timesteps=total_timesteps)
    duration = time.time() - start_time

    model_path = os.path.join(log_dir, "model.zip")
    model.save(model_path)

    with open(os.path.join(log_dir, "train_info.txt"), "w") as f:
        f.write(f"Training time (s): {duration:.2f}\n")

    rewards = wrapped_env.get_episode_rewards()
    lengths = wrapped_env.get_episode_lengths()
    stats_df = pd.DataFrame({
        "episode": np.arange(1, len(rewards) + 1),
        "ep_rew": rewards,
        "ep_len": lengths
    })
    stats_df.to_csv(os.path.join(log_dir, "rollout_stats.csv"), index=False)

    print(f"Done in {duration:.1f}s")
    print(f"Model saved to: {model_path}")
    print(f"Stats saved to: {os.path.join(log_dir, 'rollout_stats.csv')}")
    print(f"Duration saved to: {os.path.join(log_dir, 'train_info.txt')}")


if __name__ == "__main__":
    env_configs = [
        (me.EMPTY_ROOM, (5, 5)),
        (me.ROOM_WITH_MULTIPLE_MONSTERS, (7, 7))
    ]

    for env_id, chars_size in env_configs:
        train_agent(PPO, env_id, chars_size)
        train_agent(DQN, env_id, chars_size)
