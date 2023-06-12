import numpy as np
import json
import os

from ..do_not_touch.contracts import SingleAgentEnv
from ..do_not_touch.result_structures import PolicyAndActionValueFunction, Policy, ActionValueFunction
from ..do_not_touch.single_agent_env_wrapper import Env2
from ..custom_monte_carlo_env.TicTacToeEnv import TicTacToeEnv

path_save_logs = "logs/"

def argmax(dict):
    top_value = float("-inf")
    ties = []
    for key in dict.keys():
        if dict[key] > top_value:
            ties.clear()
            ties.append(key)
            top_value = dict[key]
        elif dict[key] == top_value:
            ties.append(key)
    return np.random.choice(ties)

def monte_carlo_es(env: SingleAgentEnv,
                   gamma: float = 0.99999,
                   max_episodes_count: int = 10000):
    # used for logs
    lenght_episodes = []
    reward_episodes = []

    pi: Policy = {}
    Q: ActionValueFunction = {}
    Returns = {}
    for ep_id in range(max_episodes_count):
        lenght_episode = 0

        S = []
        A = []
        R = []
        env.reset_random()
        while not env.is_game_over():
            s = env.state_id()
            aa = env.available_actions_ids()

            #initialize pi[s], Q[s] and Returns[s] if s is new
            if s not in pi.keys():
                pi[s] = {a: 1/len(aa) for a in aa}
                Q[s] = {a: np.random.uniform(-1.0, 1.0) for a in aa}
                Returns[s] = {a: [] for a in aa}

            pi_s = [pi[s][a] for a in aa]
            assert(abs(np.sum(pi_s) - 1) < 1e-9)
            a = np.random.choice(aa, p=pi_s)

            old_score = env.score()
            env.act_with_action_id(a)
            new_score = env.score()
            r = new_score - old_score
            S.append(s)
            A.append(a)
            R.append(r)
            lenght_episode += 1

        G = 0
        for t in reversed(range(len(S))):
            s_t = S[t]
            a_t = A[t]
            r_t = R[t]

            G = r_t + gamma * G
            if (s_t, a_t) not in zip(S[0: t], A[0: t]):
                Returns[s_t][a_t].append(G)
                Q[s_t][a_t] = np.mean(Returns[s_t][a_t])
                best_a = argmax(Q[s_t])
                pi[s_t] = dict.fromkeys(pi[s_t], 0)
                pi[s_t][best_a] = 1
        lenght_episodes.append(lenght_episode)
        reward_episodes.append(G)

    #save logs
    dict_logs = {
        "lenght_episodes": lenght_episodes,
        "reward_episodes": reward_episodes
    }
    with open(path_save_logs + 'monte_carlo_es_logs.json', 'w') as file:
        json.dump(dict_logs, file)

    ans: PolicyAndActionValueFunction = dict(sorted(pi.items())), dict(sorted(Q.items()))
    return ans

def on_policy_first_visit_monte_carlo_control(env: SingleAgentEnv,
                                              gamma: float = 0.99999,
                                              epsilon: float = 0.2,
                                              max_episodes_count: int = 50000):
    # used for logs
    lenght_episodes = []
    reward_episodes = []

    assert(epsilon > 0)
    pi: Policy = {}
    Q: ActionValueFunction = {}
    Returns = {}
    for ep_id in range(max_episodes_count):
        lenght_episode = 0

        S = []
        A = []
        R = []
        env.reset()
        while not env.is_game_over():
            s = env.state_id()
            aa = env.available_actions_ids()

            #initialize pi[s], Q[s] and Returns[s] if s is new
            if s not in pi.keys():
                pi[s] = {a: 1/len(aa) for a in aa}
                Q[s] = {a: np.random.uniform(-1.0, 1.0) for a in aa}
                Returns[s] = {a: [] for a in aa}

            pi_s = [pi[s][a] for a in aa]
            assert(abs(np.sum(pi_s) - 1) < 1e-9)
            a = np.random.choice(aa, p=pi_s)

            old_score = env.score()
            env.act_with_action_id(a)
            new_score = env.score()
            r = new_score - old_score
            S.append(s)
            A.append(a)
            R.append(r)
            lenght_episode += 1

        G = 0
        for t in reversed(range(len(S))):
            s_t = S[t]
            a_t = A[t]
            r_t = R[t]

            G = r_t + gamma * G
            if (s_t, a_t) not in zip(S[0: t], A[0: t]):
                Returns[s_t][a_t].append(G)
                Q[s_t][a_t] = np.mean(Returns[s_t][a_t])
                best_a = argmax(Q[s_t])
                pi[s_t] = dict.fromkeys(pi[s_t], epsilon / len(pi[s_t]))
                pi[s_t][best_a] += 1 - epsilon
        lenght_episodes.append(lenght_episode)
        reward_episodes.append(G)

    # save logs
    dict_logs = {
        "lenght_episodes": lenght_episodes,
        "reward_episodes": reward_episodes
    }
    with open(path_save_logs + 'on_policy_first_visit_monte_carlo_control_logs.json', 'w') as file:
        json.dump(dict_logs, file)

    ans: PolicyAndActionValueFunction = dict(sorted(pi.items())), dict(sorted(Q.items()))
    return ans

def off_policy_monte_carlo_control(env: SingleAgentEnv,
                                   gamma: float = 0.99999,
                                   max_episodes_count: int = 10000):
    # used for logs
    lenght_episodes = []
    reward_episodes = []

    pi: Policy = {}
    Q: ActionValueFunction = {}
    C = {}
    for ep_id in range(max_episodes_count):
        lenght_episode = 0

        b = {}
        S = []
        A = []
        R = []
        env.reset()
        while not env.is_game_over():
            s = env.state_id()
            aa = env.available_actions_ids()

            #initialize pi[s], Q[s] and Returns[s] if s is new
            if s not in pi.keys():
                Q[s] = {a: np.random.uniform(-1.0, 1.0) for a in aa}
                C[s] = {a: 0 for a in aa}
                # pi[s] = {a: 1 / len(aa) for a in aa}
                best_a = argmax(Q[s])
                pi[s] = {a: 0 for a in aa}
                pi[s][best_a] = 1
            if s not in b.keys():
                b[s] = {a: 1 / len(aa) for a in aa}

            b_s = [b[s][a] for a in aa]
            assert(abs(np.sum(b_s) - 1) < 1e-9)
            a = np.random.choice(aa, p=b_s)

            old_score = env.score()
            env.act_with_action_id(a)
            new_score = env.score()
            r = new_score - old_score
            S.append(s)
            A.append(a)
            R.append(r)
            lenght_episode += 1

        G = 0
        W = 1
        for t in reversed(range(len(S))):
            s_t = S[t]
            a_t = A[t]
            r_t = R[t]

            G = r_t + gamma * G
            C[s_t][a_t] = C[s_t][a_t] + W
            Q[s_t][a_t] = Q[s_t][a_t] + (W/C[s_t][a_t])*(G - Q[s_t][a_t])
            best_a = argmax(Q[s_t])
            pi[s_t] = dict.fromkeys(pi[s_t], 0)
            pi[s_t][best_a] = 1
            if a_t != best_a:
                break
            W = W/b[s_t][a_t]
        lenght_episodes.append(lenght_episode)
        reward_episodes.append(G)

    # save logs
    dict_logs = {
        "lenght_episodes": lenght_episodes,
        "reward_episodes": reward_episodes
    }
    with open(path_save_logs + 'off_policy_monte_carlo_control_logs.json', 'w') as file:
        json.dump(dict_logs, file)

    ans: PolicyAndActionValueFunction = dict(sorted(pi.items())), dict(sorted(Q.items()))
    return ans

def save_policy(play_first):
    pi, Q = on_policy_first_visit_monte_carlo_control_on_tic_tac_toe_solo(play_first)
    # pi, Q = off_policy_monte_carlo_control_on_tic_tac_toe_solo(play_first)
    with open(os.path.join(os.path.dirname(__file__), '../tictactoe_interface', 'policy',
                           'policy_play_first_' + str(play_first) + '.json'), 'w') as file:
        json.dump(pi, file)

def monte_carlo_es_on_tic_tac_toe_solo(play_first) -> PolicyAndActionValueFunction:
    """
    Creates a TicTacToe Solo environment (Single player versus Uniform Random Opponent)
    Launches a Monte Carlo ES (Exploring Starts) in order to find the optimal Policy and its action-value function
    Returns the Optimal Policy (Pi(s,a)) and its Action-Value function (Q(s,a))
    """
    # TODO
    env = TicTacToeEnv(play_first)
    return monte_carlo_es(env)


def on_policy_first_visit_monte_carlo_control_on_tic_tac_toe_solo(play_first) -> PolicyAndActionValueFunction:
    """
    Creates a TicTacToe Solo environment (Single player versus Uniform Random Opponent)
    Launches an On Policy First Visit Monte Carlo Control algorithm in order to find the optimal epsilon-greedy Policy
    and its action-value function
    Returns the Optimal epsilon-greedy Policy (Pi(s,a)) and its Action-Value function (Q(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    # TODO
    env = TicTacToeEnv(play_first)
    return on_policy_first_visit_monte_carlo_control(env)


def off_policy_monte_carlo_control_on_tic_tac_toe_solo(play_first) -> PolicyAndActionValueFunction:
    """
    Creates a TicTacToe Solo environment (Single player versus Uniform Random Opponent)
    Launches an Off Policy Monte Carlo Control algorithm in order to find the optimal greedy Policy and its action-value function
    Returns the Optimal Policy (Pi(s,a)) and its Action-Value function (Q(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    # TODO
    env = TicTacToeEnv(play_first)
    return off_policy_monte_carlo_control(env)


def monte_carlo_es_on_secret_env2() -> PolicyAndActionValueFunction:
    """
    Creates a Secret Env2
    Launches a Monte Carlo ES (Exploring Starts) in order to find the optimal Policy and its action-value function
    Returns the Optimal Policy (Pi(s,a)) and its Action-Value function (Q(s,a))
    """
    env = Env2()
    # TODO
    return monte_carlo_es(env)


def on_policy_first_visit_monte_carlo_control_on_secret_env2() -> PolicyAndActionValueFunction:
    """
    Creates a Secret Env2
    Launches an On Policy First Visit Monte Carlo Control algorithm in order to find the optimal epsilon-greedy Policy and its action-value function
    Returns the Optimal epsilon-greedy Policy (Pi(s,a)) and its Action-Value function (Q(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    env = Env2()
    # TODO
    return on_policy_first_visit_monte_carlo_control(env)


def off_policy_monte_carlo_control_on_secret_env2() -> PolicyAndActionValueFunction:
    """
    Creates a Secret Env2
    Launches an Off Policy Monte Carlo Control algorithm in order to find the optimal greedy Policy and its action-value function
    Returns the Optimal Policy (Pi(s,a)) and its Action-Value function (Q(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    env = Env2()
    # TODO
    return off_policy_monte_carlo_control(env)


def demo():

    # print(monte_carlo_es_on_tic_tac_toe_solo(True))
    # print(on_policy_first_visit_monte_carlo_control_on_tic_tac_toe_solo(True))
    # print(off_policy_monte_carlo_control_on_tic_tac_toe_solo(True))
    #
    # print(monte_carlo_es_on_secret_env2())
    # print(on_policy_first_visit_monte_carlo_control_on_secret_env2())
    # print(off_policy_monte_carlo_control_on_secret_env2())

    save_policy(True)
