import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import minihack_env as me
from commons import AbstractAgent, AbstractRLTask, get_crop_chars_from_observation, get_crop_pixel_from_observation
import copy

# ------------------- Q-LEARNING -------------------
class QLearningAgent(AbstractAgent):
    
    def __init__(self, id, env, action_space, epsilon=0.9, gamma=0.99, alpha=0.1):
        super().__init__(id, action_space)
        self.epsilon = epsilon
        self.gamma = gamma
        self.alpha = alpha
        self.env = env
        state, info = self.env.reset()
        size_space = get_crop_chars_from_observation(state).shape
        self.q_matrix = np.zeros((size_space[0], size_space[1], 4))

    def update(self, state, next_state, reward, action, done):
        temp = get_crop_chars_from_observation(state)
        position = np.argwhere(temp == 64)
        row, col = position[0]
        if not done:
            temp_next = get_crop_chars_from_observation(next_state)
            position_next = np.argwhere(temp_next == 64)
            row_next, col_next = position_next[0]
        else:
            row_next, col_next = temp.shape[0]-1, temp.shape[1]-1
        self.q_matrix[row, col, action] += self.alpha * (reward + (self.gamma * np.max(self.q_matrix[row_next, col_next])) - self.q_matrix[row, col, action])
    
    def select_action(self, state):
        if self.learning and np.random.rand() < self.epsilon:
            return self.action_space.sample()
        else:
            temp = get_crop_chars_from_observation(state)
            position = np.argwhere(temp == 64)
            row, col = position[0]
            return np.argmax(self.q_matrix[row, col])
     
# ------------------- RL TASK -------------------
class RLTask(AbstractRLTask):
    def __init__(self, env, agent):
        super().__init__(env, agent)

    def interact_qlearning(self, n_episodes):
        reward_episodes = []
        for episode in range(n_episodes):
            print(episode)
            reward_temp = 0
            state, info = self.env.reset()
            last_state = copy.deepcopy(state)
            while True:
                action = self.agent.select_action(last_state)
                next_state, reward, done, truncated, info = self.env.step(action)
                reward_temp += reward
                self.agent.update(last_state, next_state, reward, action, done or truncated)
                if done or truncated:
                    break
                last_state = copy.deepcopy(next_state)
            reward_episodes.append(reward_temp) 
        return reward_episodes

    def visualize_episode(self, max_number_steps=None):
        self.agent.learning = False 
        state, info = self.env.reset()
        done = False
        step_count = 0
        episode_return = 0
        plt.ion()
        fig, ax = plt.subplots()

        while not done:
            print(f"Step {step_count}:")
            self.env.render()
            if "pixel" in state:
                    ax.clear()
                    print(state.keys())
                    ax.imshow(get_crop_pixel_from_observation(state))
                    ax.set_title(f"Step {step_count}")
                    ax.axis("off")
                    plt.pause(1) 

            action = self.agent.select_action(state)
            state, reward, done, truncated, info = self.env.step(action)
            print(f"Action: {action}, Reward: {reward}")
            episode_return += reward

            step_count += 1
            print('---' * 10)
            if max_number_steps is not None and step_count >= max_number_steps:
                break

        print(f"Episode ended after {step_count} steps. Total return: {episode_return}\n")
        self.agent.onEpisodeEnd()
        self.env.close()
        plt.ioff()
        plt.show()    


def plot_rewards(rewards, method='', room = ""):
    # Compute cumulative average return
    average_returns = []
    cumulative_return = 0
    for k, episode_return in enumerate(rewards):
        cumulative_return += episode_return
        average_return_k = cumulative_return / (k + 1)
        average_returns.append(average_return_k)
    np.save(f"results_ep/{method}_{room}.npy", np.array(average_returns))
    plt.figure(figsize=(10, 5))
    plt.plot(average_returns, label='Cumulative Average Return')
    plt.xlabel("Episode")
    plt.ylabel("Average Return")
    plt.title(f"{method},{room}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    id = me.EMPTY_ROOM
    env = me.get_minihack_envirnment(id, add_pixel=True)
    agent = QLearningAgent(id, env, env.action_space)
    task = RLTask(env, agent)
    rewards = task.interact_qlearning(10000)
    plot_rewards(rewards, "QLearning", "EMPTY_big")
    task.visualize_episode()


# import numpy as np
# import matplotlib.pyplot as plt
# from collections import defaultdict
# import minihack_env as me
# from commons import AbstractAgent, AbstractRLTask, get_crop_chars_from_observation, get_crop_pixel_from_observation
# import copy

# class QLearningAgent(AbstractAgent):
#     def __init__(self, id, env, action_space, epsilon=0.1, gamma=0.99, alpha=0.1):
#         super().__init__(id, action_space)
#         self.epsilon = epsilon
#         self.gamma = gamma
#         self.alpha = alpha
#         self.env = env
#         state, info = self.env.reset()
#         self.grid_size = get_crop_chars_from_observation(state).shape 
#         n, m = self.grid_size
#         self.q_matrix = np.zeros((n, m, n + 1, m + 1, action_space.n)) 

#     def update(self, state, next_state, reward, action, done):
#         temp = get_crop_chars_from_observation(state)
#         agent_pos = np.argwhere(temp == 64)
#         monster_pos = np.argwhere(temp == 90)  

#         row, col = agent_pos[0] if agent_pos.size > 0 else (self.grid_size[0] - 1, self.grid_size[1] - 1)
#         mon_r, mon_c = monster_pos[0] if monster_pos.size > 0 else (self.grid_size[0], self.grid_size[1])

#         if not done:
#             temp_next = get_crop_chars_from_observation(next_state)
#             agent_pos_next = np.argwhere(temp_next == 64)
#             monster_pos_next = np.argwhere(temp_next == 90)

#             row_next, col_next = agent_pos_next[0] if agent_pos_next.size > 0 else (self.grid_size[0] - 1, self.grid_size[1] - 1)
#             mon_r_next, mon_c_next = monster_pos_next[0] if monster_pos_next.size > 0 else (self.grid_size[0], self.grid_size[1])
#         else:
#             row_next, col_next = temp.shape[0] - 1, temp.shape[1] - 1
#             mon_r_next, mon_c_next = self.grid_size[0], self.grid_size[1]

#         self.q_matrix[row, col, mon_r, mon_c, action] += self.alpha * (reward + self.gamma * np.max(self.q_matrix[row_next, col_next, mon_r_next, mon_c_next]) - self.q_matrix[row, col, mon_r, mon_c, action])

#     def select_action(self, state):
#         temp = get_crop_chars_from_observation(state)
#         agent_pos = np.argwhere(temp == 64)
#         monster_pos = np.argwhere(temp == 90)

#         row, col = agent_pos[0] if agent_pos.size > 0 else (self.grid_size[0] - 1, self.grid_size[1] - 1)
#         mon_r, mon_c = monster_pos[0] if monster_pos.size > 0 else (self.grid_size[0], self.grid_size[1])

#         if self.learning and np.random.rand() < self.epsilon:
#             return self.action_space.sample()
#         else:
#             return np.argmax(self.q_matrix[row, col, mon_r, mon_c])


# # ------------------- RL TASK -------------------
# class RLTask(AbstractRLTask):
#     def __init__(self, env, agent):
#         super().__init__(env, agent)

#     def interact_qlearning(self, n_episodes):
#         reward_episodes = []
#         for episode in range(n_episodes):
#             print(f"Episode {episode}")
#             reward_temp = 0
#             state, info = self.env.reset()
#             last_state = copy.deepcopy(state)
#             while True:
#                 action = self.agent.select_action(last_state)
#                 next_state, reward, done, truncated, info = self.env.step(action)
#                 reward_temp += reward
#                 self.agent.update(last_state, next_state, reward, action, done or truncated)
#                 if done or truncated:
#                     break
#                 last_state = copy.deepcopy(next_state)
#             reward_episodes.append(reward_temp)
#         return reward_episodes

#     def visualize_episode(self, max_number_steps=None):
#         self.agent.learning = False 
#         state, info = self.env.reset()
#         done = False
#         step_count = 0
#         episode_return = 0
#         plt.ion()
#         fig, ax = plt.subplots()

#         while not done:
#             print(f"Step {step_count}:")
#             self.env.render()
#             if "pixel" in state:
#                 ax.clear()
#                 ax.imshow(get_crop_pixel_from_observation(state))
#                 ax.set_title(f"Step {step_count}")
#                 ax.axis("off")
#                 plt.pause(1) 

#             action = self.agent.select_action(state)
#             state, reward, done, truncated, info = self.env.step(action)
#             print(f"Action: {action}, Reward: {reward}")
#             episode_return += reward

#             step_count += 1
#             if max_number_steps is not None and step_count >= max_number_steps:
#                 break

#         print(f"Episode ended after {step_count} steps. Total return: {episode_return}")
#         self.agent.onEpisodeEnd()
#         self.env.close()
#         plt.ioff()
#         plt.show()    


# def plot_rewards(rewards, method='', room = ""):
#     # Compute cumulative average return
#     average_returns = []
#     cumulative_return = 0
#     for k, episode_return in enumerate(rewards):
#         cumulative_return += episode_return
#         average_return_k = cumulative_return / (k + 1)
#         average_returns.append(average_return_k)
#     np.save(f"results/{method}_{room}.npy", np.array(average_returns))
#     plt.figure(figsize=(10, 5))
#     plt.plot(average_returns, label='Cumulative Average Return')
#     plt.xlabel("Episode")
#     plt.ylabel("Average Return")
#     plt.title(f"{method},{room}")
#     plt.grid(True)
#     plt.legend()
#     plt.tight_layout()
#     plt.show()

# if __name__ == "__main__":
#     id = me.ROOM_WITH_MONSTER
#     env = me.get_minihack_envirnment(id, add_pixel=True)
#     agent = QLearningAgent(id, env, env.action_space)
#     task = RLTask(env, agent)
#     rewards = task.interact_qlearning(10000)
#     plot_rewards(rewards, "QLearning", "MONSTER")
#     task.visualize_episode()