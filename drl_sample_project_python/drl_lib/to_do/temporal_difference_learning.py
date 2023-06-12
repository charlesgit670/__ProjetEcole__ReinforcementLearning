import numpy as np
import json
import os

from ..do_not_touch.result_structures import PolicyAndActionValueFunction, Policy, ActionValueFunction
from ..do_not_touch.single_agent_env_wrapper import Env3
from ..do_not_touch.contracts import SingleAgentEnv
from ..custom_monte_carlo_env.TicTacToeEnv import TicTacToeEnv
from ..to_do.monte_carlo_methods import argmax

path_save_logs = "logs/"

def sarsa(env: SingleAgentEnv,
               gamma: float = 0.9999,
               alpha: float = 0.1,
               epsilon: float = 0.2,
               max_episodes_count: int = 100000):
    assert (epsilon > 0)
    assert (alpha > 0)

    # used for logs
    lenght_episodes = []
    reward_episodes = []

    pi: Policy = {}
    Q: ActionValueFunction = {}

    for ep_id in range(max_episodes_count):
        lenght_episode = 0
        G = 0

        env.reset()
        s = env.state_id()
        aa = env.available_actions_ids()

        # initialize pi[s], Q[s] if s is new
        if s not in pi.keys():
            pi[s] = {a: 0 for a in aa}
            Q[s] = {a: np.random.uniform(-1.0, 1.0) for a in aa}

        if np.random.random() < epsilon:
            a = np.random.choice(aa)
        else:
            a = argmax(Q[s])
        while not env.is_game_over():
            old_score = env.score()
            env.act_with_action_id(a)
            new_score = env.score()
            r = new_score - old_score

            s_p = env.state_id()
            aa_p = env.available_actions_ids()
            if s_p not in pi.keys():
                pi[s_p] = {a: 0 for a in aa_p}
                Q[s_p] = {a: np.random.uniform(-1.0, 1.0) for a in aa_p}

            if env.is_game_over():
                Q[s_p] = 0.0
                Q[s][a] += alpha * (r - Q[s][a])
            else:
                if np.random.random() < epsilon:
                    a_p = np.random.choice(aa_p)
                else:
                    a_p = argmax(Q[s_p])
                Q[s][a] += alpha * (r + gamma * Q[s_p][a_p] - Q[s][a])

            pi[s] = dict.fromkeys(pi[s], 0)
            pi[s][argmax(Q[s])] = 1.0
            a = a_p
            s = s_p

            G += gamma**lenght_episode * r
            lenght_episode += 1
        lenght_episodes.append(lenght_episode)
        reward_episodes.append(G)

    # save logs
    dict_logs = {
        "lenght_episodes": lenght_episodes,
        "reward_episodes": reward_episodes
    }
    with open(path_save_logs + 'sarsa_logs.json', 'w') as file:
        json.dump(dict_logs, file)

    ans: PolicyAndActionValueFunction = pi, Q
    return ans


def q_learning(env: SingleAgentEnv,
               gamma: float = 0.9999,
               alpha: float = 0.1,
               epsilon: float = 0.2,
               max_episodes_count: int = 50000):
    assert (epsilon > 0)
    assert (alpha > 0)

    # used for logs
    lenght_episodes = []
    reward_episodes = []

    pi: Policy = {}
    Q: ActionValueFunction = {}

    for ep_id in range(max_episodes_count):
        lenght_episode = 0
        G = 0
        env.reset()
        while not env.is_game_over():
            s = env.state_id()
            aa = env.available_actions_ids()

            # initialize pi[s], Q[s] if s is new
            if s not in pi.keys():
                pi[s] = {a: 0 for a in aa}
                Q[s] = {a: np.random.uniform(-1.0, 1.0) for a in aa}

            if np.random.random() < epsilon:
                a = np.random.choice(aa)
            else:
                a = argmax(Q[s])

            old_score = env.score()
            env.act_with_action_id(a)
            new_score = env.score()
            r = new_score - old_score

            s_p = env.state_id()
            aa_p = env.available_actions_ids()
            if s_p not in pi.keys():
                pi[s_p] = {a: 0 for a in aa_p}
                Q[s_p] = {a: np.random.uniform(-1.0, 1.0) for a in aa_p}

            if env.is_game_over():
                Q[s_p] = 0.0
                Q[s][a] += alpha * (r - Q[s][a])
            else:
                Q_max = max(Q[s_p], key=lambda k: Q[s_p][k])
                Q[s][a] += alpha * (r + gamma * Q_max - Q[s][a])

            pi[s] = dict.fromkeys(pi[s], 0)
            pi[s][argmax(Q[s])] = 1.0

            G += gamma ** lenght_episode * r
            lenght_episode += 1
        lenght_episodes.append(lenght_episode)
        reward_episodes.append(G)

    # save logs
    dict_logs = {
        "lenght_episodes": lenght_episodes,
        "reward_episodes": reward_episodes
    }
    with open(path_save_logs + 'q_learning_logs.json', 'w') as file:
        json.dump(dict_logs, file)

    ans: PolicyAndActionValueFunction = pi, Q
    return ans

def expected_sarsa(env: SingleAgentEnv,
               gamma: float = 0.9999,
               alpha: float = 0.1,
               epsilon: float = 0.2,
               max_episodes_count: int = 100000):
    assert (epsilon > 0)
    assert (alpha > 0)

    # used for logs
    lenght_episodes = []
    reward_episodes = []

    pi: Policy = {}
    Q: ActionValueFunction = {}

    for ep_id in range(max_episodes_count):
        lenght_episode = 0
        G = 0
        env.reset()
        while not env.is_game_over():
            s = env.state_id()
            aa = env.available_actions_ids()

            # initialize pi[s], Q[s] if s is new
            if s not in pi.keys():
                pi[s] = {a: 0 for a in aa}
                Q[s] = {a: np.random.uniform(-1.0, 1.0) for a in aa}

            if np.random.random() < epsilon:
                a = np.random.choice(aa)
            else:
                a = argmax(Q[s])

            old_score = env.score()
            env.act_with_action_id(a)
            new_score = env.score()
            r = new_score - old_score

            s_p = env.state_id()
            aa_p = env.available_actions_ids()
            if s_p not in pi.keys():
                pi[s_p] = {a: 0 for a in aa_p}
                Q[s_p] = {a: np.random.uniform(-1.0, 1.0) for a in aa_p}

            if env.is_game_over():
                Q[s_p] = 0.0
                Q[s][a] += alpha * (r - Q[s][a])
            else:
                Q[s][a] += alpha * (r + gamma * sum(pi[s_p][a_p]*Q[s_p][a_p] for a_p in aa_p) - Q[s][a])

            pi[s] = dict.fromkeys(pi[s], 0)
            pi[s][argmax(Q[s])] = 1.0
            G += gamma ** lenght_episode * r
            lenght_episode += 1
        lenght_episodes.append(lenght_episode)
        reward_episodes.append(G)

    # save logs
    dict_logs = {
        "lenght_episodes": lenght_episodes,
        "reward_episodes": reward_episodes
    }
    with open(path_save_logs + 'expected_sarsa_logs.json', 'w') as file:
        json.dump(dict_logs, file)

    ans: PolicyAndActionValueFunction = pi, Q
    return ans

def save_policy(play_first):
    # pi, Q = q_learning_on_tic_tac_toe_solo(play_first)
    pi, Q = sarsa_on_tic_tac_toe_solo(play_first)
    # pi, Q = expected_sarsa_on_tic_tac_toe_solo(play_first)
    with open(os.path.join(os.path.dirname(__file__), '../tictactoe_interface', 'policy',
                           'policy_play_first_' + str(play_first) + '.json'), 'w') as file:
        json.dump(pi, file)

def sarsa_on_tic_tac_toe_solo(play_first) -> PolicyAndActionValueFunction:
    """
    Creates a TicTacToe Solo environment (Single player versus Uniform Random Opponent)
    Launches a SARSA Algorithm in order to find the optimal epsilon-greedy Policy and its action-value function
    Returns the optimal epsilon-greedy Policy and its Action-Value function (Q(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    # TODO
    env = TicTacToeEnv(play_first)
    return sarsa(env)


def q_learning_on_tic_tac_toe_solo(play_first) -> PolicyAndActionValueFunction:
    """
    Creates a TicTacToe Solo environment (Single player versus Uniform Random Opponent)
    Launches a Q-Learning algorithm in order to find the optimal greedy Policy and its action-value function
    Returns the optimal greedy Policy and its Action-Value function (Q(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    # TODO
    env = TicTacToeEnv(play_first)
    return q_learning(env)


def expected_sarsa_on_tic_tac_toe_solo(play_first) -> PolicyAndActionValueFunction:
    """
    Creates a TicTacToe Solo environment (Single player versus Uniform Random Opponent)
    Launches a Expected SARSA Algorithm in order to find the optimal epsilon-greedy Policy and its action-value function
    Returns the optimal epsilon-greedy Policy and its Action-Value function (Q(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    # TODO
    env = TicTacToeEnv(play_first)
    return expected_sarsa(env)


def sarsa_on_secret_env3() -> PolicyAndActionValueFunction:
    """
    Creates a Secret Env3
    Launches a SARSA Algorithm in order to find the optimal epsilon-greedy Policy and its action-value function
    Returns the optimal epsilon-greedy Policy and its Action-Value function (Q(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    env = Env3()
    # TODO
    return sarsa(env)


def q_learning_on_secret_env3() -> PolicyAndActionValueFunction:
    """
    Creates a Secret Env3
    Launches a Q-Learning algorithm in order to find the optimal greedy Policy and its action-value function
    Returns the optimal greedy Policy and its Action-Value function (Q(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    env = Env3()
    # TODO
    return q_learning(env)


def expected_sarsa_on_secret_env3() -> PolicyAndActionValueFunction:
    """
    Creates a Secret Env3
    Launches a Expected SARSA Algorithm in order to find the optimal epsilon-greedy Policy and its action-value function
    Returns the optimal epsilon-greedy Policy and its Action-Value function (Q(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    env = Env3()
    # TODO
    return expected_sarsa(env)


def demo():
    pass
    print(sarsa_on_tic_tac_toe_solo(True))
    # print(q_learning_on_tic_tac_toe_solo(True))
    # print(expected_sarsa_on_tic_tac_toe_solo(True))
    #
    # print(sarsa_on_secret_env3())
    # print(q_learning_on_secret_env3())
    # print(expected_sarsa_on_secret_env3())

    save_policy(False)
