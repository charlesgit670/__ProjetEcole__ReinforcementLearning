import os
import json
import matplotlib.pyplot as plt

def plot_reward_length(path_logs, file_name, mean_param=100):
    with open(os.path.join(path_logs, file_name), 'r') as file:
        logs = json.load(file)

        length_episodes = logs["lenght_episodes"]
        reward_episodes = logs["reward_episodes"]

        mean_lengths = [sum(length_episodes[i:i + mean_param]) / mean_param for i in range(0, len(length_episodes), mean_param) if i + mean_param <= len(length_episodes)]
        mean_rewards = [sum(reward_episodes[i:i + mean_param]) / mean_param for i in range(0, len(reward_episodes), mean_param) if i + mean_param <= len(length_episodes)]

        x_axis = list(range(len(mean_lengths)))

        # Créer une figure avec deux sous-graphiques
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Premier graphique : Mean Lengths
        axes[0].set_title(f"Mean over {mean_param} games - Lengths")
        axes[0].plot(x_axis, mean_lengths, label="Mean Lengths", color='b')
        axes[0].set_xlabel(f"Batch of {mean_param}")
        axes[0].set_ylabel("Length")
        axes[0].legend()
        axes[0].grid()

        # Deuxième graphique : Mean Rewards
        axes[1].set_title(f"Mean over {mean_param} games - Cumulative Rewards")
        axes[1].plot(x_axis, mean_rewards, label="Mean Rewards", color='r')
        axes[1].set_xlabel(f"Batch of {mean_param}")
        axes[1].set_ylabel("Rewards")
        axes[1].legend()
        axes[1].grid()

        # Affichage des plots dans une seule fenêtre
        plt.tight_layout()
        plt.show()
