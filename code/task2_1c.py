import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import minihack_env as me
from commons import AbstractAgent, AbstractRLTask, get_crop_chars_from_observation, get_crop_pixel_from_observation
import copy
import random

# ------------------- Montecarlo-LEARNING -------------------
class Montecarlo(AbstractAgent):
    def __init__(self, id, env, action_space, epsilon=0.1, gamma=0.99, alpha=0.1):
        super().__init__(id, action_space)
        self.epsilon = epsilon
        self.gamma = gamma
        self.alpha = alpha
        self.Q = defaultdict(lambda: np.zeros(self.action_space.n, dtype=np.float32))
        self._returns_sum = defaultdict(float)
        self._returns_cnt = defaultdict(int)
        self.env = env
        state, info = self.env.reset()
        size_space = get_crop_chars_from_observation(state).shape
        self.returns = {}
        self.q_matrix = np.zeros((size_space[0], size_space[1], 4))
        self.policy = np.zeros((size_space[0], size_space[1], 4))
        for i in range(self.policy.shape[0]):
            for j in range(self.policy.shape[1]):
                best_action = random.randint(0,3)
                self.policy[i,j, best_action] = 1 - epsilon +  (epsilon / 4)
                for action in range(4):
                    if action != best_action:
                        self.policy[i,j, action] =  (epsilon / 4)

    def update(self, generate_episode):
        episode_return = 0
        sa_list = []
        for state, action, reward in generate_episode:
            temp = get_crop_chars_from_observation(state)
            position = np.argwhere(temp == 64)
            row, col = position[0]
            sa_list.append((row, col, action))
        t = len(generate_episode)-1
        for i, (state, action, reward) in enumerate(reversed(generate_episode)):
            episode_return += self.gamma * episode_return + reward
            temp = get_crop_chars_from_observation(state)
            position = np.argwhere(temp == 64)
            row, col = position[0]
            if (row, col, action) not in sa_list[:t-1]:
                
                if f'{row}_{col}_{action}' in self.returns:
                    self.returns[f'{row}_{col}_{action}'].append(episode_return)
                else:
                    self.returns[f'{row}_{col}_{action}'] = [episode_return]
                self.q_matrix[row, col, action] =  np.mean(self.returns[f'{row}_{col}_{action}'])
                best_action = np.argmax(self.q_matrix[row,col])
                for action in range(4):
                    if action != best_action:
                        self.policy[row,col, action] = (self.epsilon / 4)
                    else:
                        self.policy[row,col, best_action] = 1 - self.epsilon +  (self.epsilon / 4)
            t -= 1
    def select_action(self, state):
        temp = get_crop_chars_from_observation(state)
        position = np.argwhere(temp == 64)
        row, col = position[0]
        action = np.random.choice(len(self.policy[row, col]), p=self.policy[row, col])
        return action
     
# ------------------- RL TASK -------------------
class RLTask(AbstractRLTask):
    def __init__(self, env, agent):
        super().__init__(env, agent)

    def interact_mc(self, n_episodes):
        reward_episodes = []
        for episode in range(n_episodes):
            generated_episode = []
            print(episode)
            reward_temp = 0
            state, info = self.env.reset()
            last_state = copy.deepcopy(state)
            while True:
                action = self.agent.select_action(last_state)
                next_state, reward, done, truncated, info = self.env.step(action)
                reward_temp += reward
                generated_episode.append((last_state,action,reward))
                last_state = copy.deepcopy(next_state)
                if done or truncated:
                    break
            self.agent.update(generated_episode)
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

def plot_rewards(rewards, method='', room = ""):
    average_returns = []
    cumulative_return = 0
    for k, episode_return in enumerate(rewards):
        cumulative_return += episode_return
        average_return_k = cumulative_return / (k + 1)
        average_returns.append(average_return_k)
    np.save(f"results/{method}_{room}.npy", np.array(average_returns))
    plt.figure(figsize=(10, 5))
    plt.plot(average_returns, label='Cumulative Average Return')
    plt.xlabel("Episode")
    plt.ylabel("Average Return")
    plt.title(f"{method},{room}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
# ------------------- MAIN -------------------
if __name__ == "__main__":
    id = me.EMPTY_ROOM
    env = me.get_minihack_envirnment(id, add_pixel=True)
    agent = Montecarlo(me.EMPTY_ROOM, env, env.action_space)
    task = RLTask(env, agent)
    rewards = task.interact_mc(10_000)
    plot_rewards(rewards, "Montecarlo", "EMPTY")
    task.visualize_episode()


# import numpy as np
# import matplotlib.pyplot as plt
# from collections import defaultdict
# import minihack_env as me
# from commons import AbstractAgent, AbstractRLTask, get_crop_chars_from_observation, get_crop_pixel_from_observation
# import copy


# # ------------------- MONTE CARLO AGENT (WITH MONSTER + POLICY MATRIX) -------------------
# class Montecarlo(AbstractAgent):
#     def __init__(self, id, env, action_space, epsilon=0.1, gamma=0.99):
#         super().__init__(id, action_space)
#         self.epsilon = epsilon
#         self.gamma = gamma
#         self.env = env

#         state, info = self.env.reset()
#         self.grid_size = get_crop_chars_from_observation(state).shape
#         n, m = self.grid_size

#         self.q_matrix = np.zeros((n, m, n + 1, m + 1, action_space.n))
#         self.policy = np.full((n, m, n + 1, m + 1, action_space.n), fill_value=1.0 / action_space.n)
#         self.returns = defaultdict(list)

#     def _get_positions(self, state):
#         temp = get_crop_chars_from_observation(state)
#         agent_pos = np.argwhere(temp == 64)
#         monster_pos = np.argwhere(temp == 90)
#         row, col = agent_pos[0] if agent_pos.size > 0 else (self.grid_size[0] - 1, self.grid_size[1] - 1)
#         mon_r, mon_c = monster_pos[0] if monster_pos.size > 0 else (self.grid_size[0], self.grid_size[1])
#         return row, col, mon_r, mon_c

#     def update(self, episode):
#         G = 0
#         visited = set()
#         for t in reversed(range(len(episode))):
#             state, action, reward = episode[t]
#             G = self.gamma * G + reward
#             row, col, mon_r, mon_c = self._get_positions(state)
#             key = (row, col, mon_r, mon_c, action)
#             if key not in visited:
#                 self.returns[key].append(G)
#                 self.q_matrix[row, col, mon_r, mon_c, action] = np.mean(self.returns[key])
#                 visited.add(key)

#                 # Update epsilon-soft policy
#                 best_action = np.argmax(self.q_matrix[row, col, mon_r, mon_c])
#                 for a in range(self.action_space.n):
#                     if a == best_action:
#                         self.policy[row, col, mon_r, mon_c, a] = 1 - self.epsilon + self.epsilon / self.action_space.n
#                     else:
#                         self.policy[row, col, mon_r, mon_c, a] = self.epsilon / self.action_space.n

#     def select_action(self, state):
#         row, col, mon_r, mon_c = self._get_positions(state)
#         probs = self.policy[row, col, mon_r, mon_c]
#         probs /= probs.sum()  # Normalize in case of floating-point errors
#         return np.random.choice(self.action_space.n, p=probs)


# # ------------------- RL TASK -------------------
# class RLTask(AbstractRLTask):
#     def __init__(self, env, agent):
#         super().__init__(env, agent)

#     def interact_mc(self, n_episodes):
#         reward_episodes = []
#         for episode in range(n_episodes):
#             print(f"Episode {episode}")
#             episode_data = []
#             state, info = self.env.reset()
#             last_state = copy.deepcopy(state)
#             reward_temp = 0

#             while True:
#                 action = self.agent.select_action(last_state)
#                 next_state, reward, done, truncated, info = self.env.step(action)
#                 reward_temp += reward
#                 episode_data.append((last_state, action, reward))
#                 last_state = copy.deepcopy(next_state)
#                 if done or truncated:
#                     break

#             self.agent.update(episode_data)
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


# # ------------------- PLOT FUNCTION -------------------
# def plot_rewards(rewards, method='', room=""):
#     average_returns = []
#     cumulative_return = 0
#     for k, episode_return in enumerate(rewards):
#         cumulative_return += episode_return
#         average_returns.append(cumulative_return / (k + 1))
#     np.save(f"results/{method}_{room}.npy", np.array(average_returns))
#     plt.figure(figsize=(10, 5))
#     plt.plot(average_returns, label='Cumulative Average Return')
#     plt.xlabel("Episode")
#     plt.ylabel("Average Return")
#     plt.title(f"{method}, {room}")
#     plt.grid(True)
#     plt.legend()
#     plt.tight_layout()
#     plt.show()


# # ------------------- MAIN -------------------
# if __name__ == "__main__":
#     id = me.ROOM_WITH_MONSTER
#     env = me.get_minihack_envirnment(id, add_pixel=True)
#     agent = Montecarlo(id, env, env.action_space)
#     task = RLTask(env, agent)
#     rewards = task.interact_mc(10_000)
#     plot_rewards(rewards, "Montecarlo", "MONSTER")
#     task.visualize_episode()
