import os
import json
import matplotlib.pyplot as plt

def plot_reward_lengh(path_logs, file_name, mean_param=100):
    with open(os.path.join(path_logs, file_name), 'r') as file:
        logs = json.load(file)

        lenght_episodes = logs["lenght_episodes"]
        reward_episodes = logs["reward_episodes"]

        mean_lenghts = [sum(lenght_episodes[i:i + mean_param]) / mean_param for i in range(0, len(lenght_episodes), mean_param) if i + mean_param <= len(lenght_episodes)]
        mean_rewards = [sum(reward_episodes[i:i + mean_param]) / mean_param for i in range(0, len(reward_episodes), mean_param) if i + mean_param <= len(lenght_episodes)]

        x_axis = [i for i in range(0, len(mean_lenghts))]
        fig, ax = plt.subplots()
        ax.set_title(f"mean over {mean_param} points lenghts")
        ax.plot(x_axis, mean_lenghts)

        fig2, ax2 = plt.subplots()
        ax2.set_title(f"mean over {mean_param} points for cumulative rewards")
        ax2.plot(x_axis, mean_rewards)

        plt.show()
