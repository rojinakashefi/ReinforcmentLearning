import gymnasium as gym
from gymnasium import spaces
import numpy as np
from commons import AbstractAgent
from commons import AbstractRLTask
import matplotlib.pyplot as plt
import time
import pygame


class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human"]}
    def __init__(self, n=5, m=5):
        super().__init__()
        self.n = n
        self.m = m
        self.observation_space = spaces.Box(low=0, high=max(n, m), shape=(2,), dtype=np.int32)
        self.action_space = spaces.Discrete(4)  # 0: up, 1: down, 2: left, 3: right

        self.action_map = {
            0: (-1, 0),  # up
            1: (1, 0),   # down
            2: (0, -1),  # left
            3: (0, 1),   # right
        }

        self.agent_pos = np.array([0, 0], dtype=np.int32)
        self.cell_size = 60 
        self.window_size = (self.m * self.cell_size, self.n * self.cell_size)
        self.screen = None
        self.clock = None
        self.render_initialized = False


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.agent_pos = np.array([0, 0], dtype=np.int32)
        observation = self.agent_pos.copy()
        info = {}
        return observation, info

    
    def step(self, action):
        move = self.action_map[action]
        proposed_pos = self.agent_pos + np.array(move)

        if 0 <= proposed_pos[0] < self.n and 0 <= proposed_pos[1] < self.m:
            self.agent_pos = proposed_pos

        observation = self.agent_pos.copy()

        reward = -1

        goal_pos = np.array([self.n - 1, self.m - 1])
        done = np.array_equal(self.agent_pos, goal_pos)

        info = {}
        return observation, reward, done, False, info

    def render(self, mode="human"):
        if not self.render_initialized:
            pygame.init()
            self.screen = pygame.display.set_mode(self.window_size)
            pygame.display.set_caption("GridWorld")
            self.clock = pygame.time.Clock()
            self.render_initialized = True

        WHITE = (255, 255, 255)
        BLACK = (0, 0, 0)
        BLUE = (0, 0, 255)    
        GREEN = (0, 255, 0)  

        self.screen.fill(WHITE)

        for i in range(self.n):
            for j in range(self.m):
                rect = pygame.Rect(j * self.cell_size, i * self.cell_size, self.cell_size, self.cell_size)
                pygame.draw.rect(self.screen, BLACK, rect, 1)

        goal_x, goal_y = self.n - 1, self.m - 1
        goal_rect = pygame.Rect(goal_y * self.cell_size, goal_x * self.cell_size, self.cell_size, self.cell_size)
        pygame.draw.rect(self.screen, GREEN, goal_rect)

        x, y = self.agent_pos
        agent_rect = pygame.Rect(y * self.cell_size, x * self.cell_size, self.cell_size, self.cell_size)
        pygame.draw.rect(self.screen, BLUE, agent_rect)

        pygame.display.flip()
        self.clock.tick(10) 
    def close(self):
        if self.render_initialized:
            pygame.quit()
            self.render_initialized = False

class RandomAgent(AbstractAgent):
    def __init__(self, id, action_space):
        super().__init__(id, action_space)

    def act(self, state, reward=-1):
        return self.action_space.sample()
    
    def onEpisodeEnd(self, *args, **kwargs):
        pass

class RLTask(AbstractRLTask):
    def __init__(self, env, agent):
        super().__init__(env, agent)

    def interact(self, n_episodes):
        average_returns = []
        cumulative_return = 0

        for k in range(n_episodes):
            state, info = self.env.reset()
            done = False
            episode_return = 0

            while not done:
                action = self.agent.act(state)
                state, reward, done, truncated, info = self.env.step(action)
                episode_return += reward

            cumulative_return += episode_return
            average_return_k = cumulative_return / (k + 1)
            average_returns.append(average_return_k)

            self.agent.onEpisodeEnd()

        return average_returns

    def visualize_episode(self, max_number_steps=None):
        self.agent.learning = False 
        state, info = self.env.reset()
        done = False
        step_count = 0

        while not done:
            print(f"Step {step_count}:")
            self.env.render()
            action = self.agent.act(state)
            state, reward, done, truncated, info = self.env.step(action)
            print(f"   Action: {action}, Reward: {reward}")
            step_count += 1
            time.sleep(2)
            print('---'*10)
            if max_number_steps is not None and step_count >= max_number_steps:
                break
        self.agent.onEpisodeEnd()
        self.env.close()


# -------------------------------
# Grid world test without pygame

# env = GridWorldEnv(n=5, m=5)
# # Reset the environment
# obs, info = env.reset()
# env.render()
# # Define a simple policy (random actions or fixed sequence)
# actions = [3, 3, 1, 1, 2, 1, 0, 3] 
# for action in actions:
#     obs, reward, done, _, info = env.step(action)
#     env.render()
#     print(f"Action: {action}, Observation: {obs}, Reward: {reward}, Done: {done}")
#     time.sleep(0.5)  # Pause to visualize steps
#     if done:
#         print("Reached the goal!")
#         break
# env.close()
# -------------------------------
# Grid world test with pygame
# env = GridWorldEnv(n=5, m=5)
# obs, info = env.reset()
# # Define a sequence of actions (e.g., right → right → down → down → etc.)
# actions = [3, 3, 1, 1, 2, 1, 0, 3] 
# # Run the sequence
# for action in actions:
#     obs, reward, done, _, _ = env.step(action)
#     env.render()
#     print(f"Action: {action}, Obs: {obs}, Reward: {reward}, Done: {done}")
#     time.sleep(2)  # Short pause for visible updates

#     if done:
#         print("Reached the goal!")
#         break
# env.close()
# -------------------------------
# Random agent test without and with pygame
# env = GridWorldEnv(n=5, m=5)
# agent = RandomAgent("agent1", env.action_space)
# obs, info = env.reset()
# done = False
# while not done:
#     action = agent.act(obs)
#     obs, reward, done, _, _ = env.step(action)
#     env.render()
# -------------------------------
# RLTask test with pygame and without pygame
# env = GridWorldEnv(n=5, m=5)
# agent = RandomAgent("random", env.action_space)
# task = RLTask(env, agent)
# avg_returns = task.interact(10000)
# plt.plot(avg_returns)
# plt.xlabel("Episode")
# plt.ylabel("Average Return")
# plt.title("Average Return over Episodes")
# plt.show()
# task.visualize_episode(max_number_steps=10)
