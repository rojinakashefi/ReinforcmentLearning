# ------------ Task 2.1 ------------
# import numpy as np
# import matplotlib.pyplot as plt
# # Environments and methods
# environments = ["EMPTY", "LAVA", "CLIFF", "MONSTER"]
# methods = ["QLearning", "Sarsa", "MonteCarlo"]
# labels = {"QLearning": "Q-Learning", "Sarsa": "SARSA", "MonteCarlo": "Monte Carlo"}
# colors = {"QLearning": "tab:blue", "Sarsa": "tab:orange", "MonteCarlo": "tab:green"}

# # Create 2x2 subplots
# fig, axs = plt.subplots(2, 2, figsize=(14, 10))
# axs = axs.flatten()

# for idx, env in enumerate(environments):
#     ax = axs[idx]
#     for method in methods:
#         filepath = f"results/{method}_{env}.npy"
#         try:
#             data = np.load(filepath)
#             episodes = np.arange(1, len(data) + 1)
#             ax.plot(episodes, data, label=labels[method], color=colors[method])
#         except FileNotFoundError:
#             print(f"File not found: {filepath}")
#             continue
    
#     ax.set_title(f"{env} Environment")
#     ax.set_xlabel("Episode")
#     ax.set_ylabel("Average Return")
#     ax.grid(True)
#     ax.legend()

# plt.suptitle("Monte Carlo vs Q-Learning vs SARSA Across Environments", fontsize=16)
# plt.tight_layout(rect=[0, 0.03, 1, 0.95])
# plt.subplots_adjust(hspace=0.4)  # Add vertical space between rows
# plt.show()

# ------------ Task 2.1 Learning rate ------------

# import numpy as np
# import matplotlib.pyplot as plt

# methods = ["QLearning", "Sarsa"]
# learning_rates = {
#     "1e-5": "results_lr/{method}_EMPTY_small.npy",
#     "0.1": "results/{method}_EMPTY.npy",
#     "100": "results_lr/{method}_EMPTY_big.npy"
# }
# colors = {"1e-5": "tab:blue", "0.1": "tab:orange", "100": "tab:green"}

# fig, axs = plt.subplots(1, 2, figsize=(20, 5))

# for i, method in enumerate(methods):
#     ax = axs[i]
#     for lr, path_template in learning_rates.items():
#         path = path_template.format(method=method)
#         try:
#             data = np.load(path)
#             episodes = np.arange(1, len(data) + 1)
#             ax.plot(episodes, data, label=f"lr = {lr}", color=colors[lr])
#         except FileNotFoundError:
#             print(f"Missing file: {path}")
#             continue
#     ax.set_title(f"{method} in EMPTY Room")
#     ax.set_xlabel("Episode")
#     ax.set_ylabel("Average Return")
#     ax.grid(True)
#     ax.legend()

# plt.suptitle("Learning Rate Comparison for Each Method (EMPTY Room)", fontsize=16)
# plt.tight_layout(rect=[0.03, 0.03, 1, 0.95], pad=3.0)
# # Save the figure
# output_path = "output.png"
# plt.savefig(output_path)
# plt.show()


# ------------ Task 2.1 Epsilon ------------

# import numpy as np
# import matplotlib.pyplot as plt

# methods = ["QLearning", "Sarsa", "MonteCarlo"]
# learning_rates = {
#     "0.01": "results_ep/{method}_EMPTY_small.npy",
#     "0.1": "results/{method}_EMPTY.npy",
#     "0.9": "results_ep/{method}_EMPTY_big.npy"
# }
# colors = {"0.01": "tab:blue", "0.1": "tab:orange", "0.9": "tab:green"}

# fig, axs = plt.subplots(1, 3, figsize=(20, 5))

# for i, method in enumerate(methods):
#     ax = axs[i]
#     for lr, path_template in learning_rates.items():
#         path = path_template.format(method=method)
#         try:
#             data = np.load(path)
#             episodes = np.arange(1, len(data) + 1)
#             ax.plot(episodes, data, label=f"epsilon = {lr}", color=colors[lr])
#         except FileNotFoundError:
#             print(f"Missing file: {path}")
#             continue
#     ax.set_title(f"{method} in EMPTY Room")
#     ax.set_xlabel("Episode")
#     ax.set_ylabel("Average Return")
#     ax.grid(True)
#     ax.legend()

# plt.suptitle("Epsilon Rate Comparison for Each Method (EMPTY Room)", fontsize=16)
# plt.tight_layout(rect=[0.03, 0.03, 1, 0.95], pad=3.0)
# # Save the figure
# output_path = "output.png"
# plt.savefig(output_path)
# plt.show()


# ------------ Task 2.3 ------------

# import numpy as np
# import matplotlib.pyplot as plt

# # Environments and methods
# environments = ["EMPTY", "CLIFF"]
# methods = ["QLearning", "DynaQ"]
# labels = {"QLearning": "Q-Learning", "DynaQ": "DynaQ"}
# colors = {"QLearning": "tab:blue", "DynaQ": "tab:orange"}

# # Create 2x2 subplots
# fig, axs = plt.subplots(1, 2, figsize=(10, 5))
# axs = axs.flatten()

# for idx, env in enumerate(environments):
#     ax = axs[idx]
#     for method in methods:
#         filepath = f"results/{method}_{env}.npy"
#         try:
#             data = np.load(filepath)
#             episodes = np.arange(1, len(data) + 1)
#             ax.plot(episodes, data, label=labels[method], color=colors[method])
#         except FileNotFoundError:
#             print(f"File not found: {filepath}")
#             continue
    
#     ax.set_title(f"{env} Environment")
#     ax.set_xlabel("Episode")
#     ax.set_ylabel("Average Return")
#     ax.grid(True)
#     ax.legend()

# plt.suptitle("Monte Carlo vs Q-Learning vs SARSA Across Environments", fontsize=16)
# plt.tight_layout(rect=[0, 0.03, 1, 0.95])
# plt.subplots_adjust(hspace=0.4)  # Add vertical space between rows
# plt.show()


# ------------ Task 3.2 ------------ 

# import pandas as pd
# import matplotlib.pyplot as plt

# # Load your CSV
# # df = pd.read_csv("/Users/rojina/Desktop/RL/results_dl/dqn_empty-room_5x5/rollout_stats.csv") 
# df = pd.read_csv("/Users/rojina/Desktop/RL/results_dl/ppo_empty-room_5x5/rollout_stats.csv") 
# # df = pd.read_csv("/Users/rojina/Desktop/RL/results_dl/dqn_room-with-multiple-monsters_7x7/rollout_stats.csv") 
# # df = pd.read_csv("/Users/rojina/Desktop/RL/results_dl/ppo_room-with-multiple-monsters_7x7/rollout_stats.csv") 

# window = 200
# df['ep_rew_smooth'] = df['ep_rew'].rolling(window=window).mean()
# df['ep_len_smooth'] = df['ep_len'].rolling(window=window).mean()

# # Drop rows with NaN from smoothing
# df_smooth = df.dropna(subset=['ep_rew_smooth', 'ep_len_smooth'])


# # Only keep first 30,000 episodes
# df_smooth = df_smooth[df_smooth['episode'] <= 30000]

# # Create subplots in one row
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))  # 1 row, 2 columns

# # Plot 1: Episode Length
# ax1.plot(df_smooth['episode'], df_smooth['ep_len_smooth'], linewidth=2, color='steelblue')
# ax1.set_title("Episode Length")
# ax1.set_xlabel("Number of Episode")
# ax1.set_ylabel("Episode Length")

# # Plot 2: Episode Reward
# ax2.plot(df_smooth['episode'], df_smooth['ep_rew_smooth'], linewidth=2, color='darkorange')
# ax2.set_title("Episode Reward")
# ax2.set_xlabel("Number of Episode")
# ax2.set_ylabel("Episode Reward")

# # Layout adjustment
# plt.tight_layout()
# plt.show()

# ------------ Task 3.3 ------------ 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

q_table_path = "results/QLearning_EMPTY.npy"
q_table_data = np.load(q_table_path)
episodes_q = np.arange(1, len(q_table_data) + 1)
q_table_data = q_table_data[:10000]
episodes_q = episodes_q[:10000]
df_dqn = pd.read_csv("/Users/rojina/Desktop/RL/results_dl/dqn_empty-room_5x5/rollout_stats.csv")
df_dqn = df_dqn[df_dqn['episode'] <= 10000]
df_dqn['ep_rew_smooth'] = df_dqn['ep_rew'].rolling(window=200).mean()
df_dqn_smooth = df_dqn.dropna(subset=['ep_rew_smooth'])

plt.figure(figsize=(10, 6))

plt.plot(episodes_q, q_table_data, label="Q-Learning (Tabular)", color='tab:blue', linewidth=2)

plt.plot(df_dqn_smooth['episode'], df_dqn_smooth['ep_rew_smooth'], label="DQN", color='tab:orange', linewidth=2)

plt.title("Tabular Q-Learning vs DQN vs PPO in Empty Room")
plt.xlabel("Episode")
plt.ylabel("Average Return")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
