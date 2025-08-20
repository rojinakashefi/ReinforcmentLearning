import time
import numpy as np
import matplotlib.pyplot as plt
import minihack_env as me
from commons import AbstractAgent
from commons import AbstractRLTask
from commons import get_crop_chars_from_observation, get_crop_pixel_from_observation

class FixedAgent(AbstractAgent):
    def __init__(self, id, action_space, grid_height=5):
        super().__init__(id, action_space)
        self.grid_height = grid_height
# 0 up, 2 down, 3 left, 1 right
    def act(self, state, reward=-1):
            temp = get_crop_chars_from_observation(state) 
            position = np.argwhere(temp == 64)
            row, col = position[0]
            if row + 1 < temp.shape[0] and temp[row + 1, col] != 45:
                return 2
            else:
                return 1

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
                    plt.pause(2) 

            action = self.agent.act(state)
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

def run_fixed_agent_visualization(env_id, agent_id, grid_height):
    print(f"\n Visualizing {agent_id} on {env_id}")
    env = me.get_minihack_envirnment(env_id, add_pixel=True)
    agent = FixedAgent(agent_id, env.action_space, grid_height=grid_height)
    task = RLTask(env, agent)
    task.visualize_episode(max_number_steps=10)

if __name__ == "__main__":
    run_fixed_agent_visualization(me.EMPTY_ROOM, "FixedAgent-EmptyRoom", 5)
    run_fixed_agent_visualization(me.ROOM_WITH_LAVA, "FixedAgent-RoomWithLava", 5)
