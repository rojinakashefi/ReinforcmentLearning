import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import minihack_env as me
from commons import AbstractAgent, AbstractRLTask, get_crop_chars_from_observation, get_crop_pixel_from_observation
import copy

# ------------------- Q-LEARNING -------------------
class QLearningAgent(AbstractAgent):
    def __init__(self, id, env, action_space, epsilon=0.1, gamma=0.99, alpha=0.1):
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
     
class RLTask(AbstractRLTask):
    def __init__(self, env, agent):
        super().__init__(env, agent)
    def interact_qlearning(self, n_episodes, epsilon_start=1.0, epsilon_end=0.01):
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
            self.agent.epsilon = max(
                epsilon_end,
                epsilon_start - episode * (epsilon_start - epsilon_end) / 200
            )
        plt.plot(reward_episodes)
        plt.show()

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
                    print('Here')
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

if __name__ == "__main__":
    id = me.CLIFF
    env = me.get_minihack_envirnment(id, add_pixel=True)
    agent = QLearningAgent(id, env, env.action_space)
    task = RLTask(env, agent)
    task.interact_qlearning(20, epsilon_start=1.0, epsilon_end=0.01)
    task.visualize_episode()

# import numpy as np
# import matplotlib.pyplot as plt
# from collections import defaultdict
# import minihack_env as me
# from commons import AbstractAgent, AbstractRLTask, get_crop_chars_from_observation, get_crop_pixel_from_observation
# import copy


# # ------------------- SARSA -------------------
# class Sarsa(AbstractAgent):
#     def __init__(self, id, env, action_space, epsilon=0.1, gamma=0.99, alpha=0.1):
#         super().__init__(id, action_space)
#         self.epsilon = epsilon
#         self.gamma = gamma
#         self.alpha = alpha
#         self.env = env
#         state, info = self.env.reset()
#         size_space = get_crop_chars_from_observation(state).shape
#         self.q_matrix = np.zeros((size_space[0], size_space[1], 4))
#         # import ipdb; ipdb.set_trace()
#         # print('here')
#     def update(self, state, next_state, reward, action, next_action,  done):
#         temp = get_crop_chars_from_observation(state)
#         position = np.argwhere(temp == 64)
#         # import ipdb; ipdb.set_trace()
#         row, col = position[0]
#         if not done:
#             temp_next = get_crop_chars_from_observation(next_state)
#             position_next = np.argwhere(temp_next == 64)
#             row_next, col_next = position_next[0]
#         else:
#             #print('here')
#             row_next, col_next = temp.shape[0]-1, temp.shape[1]-1
#             #print(row_next, col_next)
#         self.q_matrix[row, col, action] += self.alpha * (reward + (self.gamma * self.q_matrix[row_next, col_next, next_action]) - self.q_matrix[row,col,action])
    
#     def select_action(self, state):
#         if self.learning and np.random.rand() < self.epsilon:
#             return self.action_space.sample()
#         else:
#             temp = get_crop_chars_from_observation(state)
#             position = np.argwhere(temp == 64)
#             row, col = position[0]
#             return np.argmax(self.q_matrix[row, col])
     
# # ------------------- RL TASK -------------------
# class RLTask(AbstractRLTask):
#     def __init__(self, env, agent):
#         super().__init__(env, agent)
#     def interact_sarsa(self, n_episodes, epsilon_start=1.0, epsilon_end=0.1):
#         reward_episodes = []
#         for episode in range(n_episodes):
#             print(episode)
#             reward_temp = 0
#             state, info = self.env.reset()
#             last_state = copy.deepcopy(state)
#             action = self.agent.select_action(last_state)
#             while True:
#                 next_state, reward, done, truncated, info = self.env.step(action)
#                 if done or truncated:
#                     break
#                 reward_temp += reward
#                 next_action = self.agent.select_action(next_state)
#                 self.agent.update(last_state, next_state, reward, action, next_action, done or truncated)
#                 last_state = copy.deepcopy(next_state)
#                 action = next_action
#             reward_episodes.append(reward_temp)

#             self.agent.epsilon = max(
#                 epsilon_end,
#                 epsilon_start - episode * (epsilon_start - epsilon_end) / 200
#             )
#         plt.plot(reward_episodes)
#         plt.show()

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
#                     print('Here')
#                     ax.clear()
#                     print(state.keys())
#                     ax.imshow(get_crop_pixel_from_observation(state))
#                     ax.set_title(f"Step {step_count}")
#                     ax.axis("off")
#                     plt.pause(1) 

#             action = self.agent.select_action(state)
#             state, reward, done, truncated, info = self.env.step(action)
#             print(f"Action: {action}, Reward: {reward}")
#             episode_return += reward

#             step_count += 1
#             print('---' * 10)
#             if max_number_steps is not None and step_count >= max_number_steps:
#                 break

#         print(f"Episode ended after {step_count} steps. Total return: {episode_return}\n")
#         self.agent.onEpisodeEnd()
#         self.env.close()
#         plt.ioff()
#         plt.show()    


# # ------------------- MAIN -------------------
# if __name__ == "__main__":
#     id = me.CLIFF
#     env = me.get_minihack_envirnment(id, add_pixel=True)
#     agent = Sarsa(id, env, env.action_space)
#     task = RLTask(env, agent)
#     task.interact_sarsa(n_episodes=20, epsilon_start=1.0, epsilon_end=0.01)
#     print(agent.q_matrix)
#     task.visualize_episode()


