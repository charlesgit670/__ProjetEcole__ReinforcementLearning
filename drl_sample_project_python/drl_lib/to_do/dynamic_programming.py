import numpy as np
import random
import seaborn as sns
import matplotlib.pyplot as plt
import pprint

from ..do_not_touch.mdp_env_wrapper import Env1
from ..do_not_touch.result_structures import ValueFunction, Policy, PolicyAndValueFunction
from ..custom_dynamic_programming_env.LineWorld_utils import env_line_world, p_line_world, pi_random_line_world
from ..custom_dynamic_programming_env.GridWorld_utils import env_grid_world, p_grid_world, pi_random_grid_world

def argmax(list):
    top_value = float("-inf")
    ties = []
    for i in range(len(list)):
        if list[i] > top_value:
            ties.clear()
            ties.append(i)
            top_value = list[i]
        if list[i] == top_value:
            ties.append(i)
    return np.random.choice(ties)

def policy_evaluation(S, A, R, p, pi):
    theta = 0.0000001
    V: ValueFunction = {S[i]: 0 for i in range(len(S))}

    while True:
        delta = 0.0
        for s in S:
            old_v = V[s]
            total = 0.0
            for a in A:
                total_inter = 0.0
                for s_p in S:
                    for r in range(len(R)):
                        total_inter += p(s, a, s_p, r) * (R[r] + 0.99999 * V[s_p])
                total_inter = pi(s, a) * total_inter
                total += total_inter
            V[s] = total
            delta = max(delta, np.abs(V[s] - old_v))
        if delta < theta:
            return V

def policy_iteration(S, A, R, p):
    theta = 0.0000001

    pi_random = [[random.random()/len(A) for _ in range(len(A))] for _ in range(len(S))]
    pi: Policy = {S[i]:
                      {A[j]: pi_random[i][j] for j in range(len(A))}
                  for i in range(len(S))}
    V: ValueFunction = {S[i]: 0 for i in range(len(S))}

    while True:
        # Policy Evaluation
        while True:
            delta = 0.0
            for s in S:
                old_v = V[s]
                total = 0.0
                a = max(pi[s], key=lambda k: pi[s][k])
                for s_p in S:
                    for r in range(len(R)):
                        total += p(s, a, s_p, r) * (R[r] + 0.99999 * V[s_p])
                V[s] = total
                delta = max(delta, np.abs(V[s] - old_v))
            if delta < theta:
                break

        # Policy Improvement
        policy_stable = True
        for s in S:
            old_a = max(pi[s], key=lambda k: pi[s][k])
            # argmax from scratch (si argmax random, possible boucle infinie car on a plusieurs pi optimal)
            best_a = None
            best_a_score = None
            for a in A:
                total = 0.0
                for s_p in S:
                    for r in range(len(R)):
                        total += p(s, a, s_p, r) * (R[r] + 0.99999 * V[s_p])
                if best_a is None or total > best_a_score:
                    best_a = a
                    best_a_score = total

            pi[s] = dict.fromkeys(pi[s], 0)
            pi[s][best_a] = 1
            if old_a != best_a:
                policy_stable = False
        if policy_stable:
            break

        ans: PolicyAndValueFunction = pi, V
    return ans

def value_iteration(S, A, R, p):
    theta = 0.0000001

    V: ValueFunction = {S[i]: 0 for i in range(len(S))}

    while True:
        delta = 0.0
        for s in S:
            old_v = V[s]
            best_value = None
            for a in A:
                total = 0.0
                for s_p in S:
                    for r in range(len(R)):
                        total += p(s, a, s_p, r) * (R[r] + 0.99999 * V[s_p])
                if best_value is None or total > best_value:
                    best_value = total
            V[s] = best_value
            delta = max(delta, np.abs(V[s] - old_v))
        if delta < theta:
            break

    pi: Policy = {}
    for s in S:
        values = []
        for a in A:
            total = 0.0
            for s_p in S:
                for r in range(len(R)):
                    total += p(s, a, s_p, r) * (R[r] + 0.99999 * V[s_p])
            values.append(total)
        pi[s] = {A[j]: 0 for j in A}
        pi[s][argmax(values)] = 1

    ans: PolicyAndValueFunction = pi, V

    return ans

def plot_grid_world(pi, V):
    matrix = [[1 if V.get((x, y)) == 0 else V.get((x, y)) for y in range(5)] for x in range(5)]


    fig, ax = plt.subplots()
    sns.heatmap(matrix, cmap='coolwarm', cbar=True, ax=ax)

    ax.set_xlim(0, 5)
    ax.set_ylim(5, 0)

    if pi is not None:
        directions = {key: max(value, key=value.get) for key, value in pi.items() if len(value) > 0}
        # Parcours de la grille et des directions pour tracer les flèches
        for (x, y), direction in directions.items():
            dx, dy = 0, 0
            if direction == 0:
                dx = -0.4
            elif direction == 1:
                dx = 0.4
            elif direction == 2:
                dy = 0.4
            elif direction == 3:
                dy = -0.4
            ax.arrow(y + 0.5, x + 0.5, dx, dy, head_width=0.1, head_length=0.1, fc='black')
    plt.show()

def plot_line_world(pi, V):
    matrix = [[1 if V.get(y) == 0 else V.get(y) for y in range(len(V))] for x in range(1)]
    directions = {key: max(value, key=value.get) for key, value in pi.items()}

    fig, ax = plt.subplots(figsize=(8, 2))
    sns.heatmap(matrix, cmap='coolwarm', cbar=True, ax=ax)

    ax.set_xlim(0, len(V))
    ax.set_ylim(0, 1)

    # Parcours de la grille et des directions pour tracer les flèches
    for x, direction in directions.items():
        dx, dy = 0, 0
        if direction == 0:
            dx = -0.4
        elif direction == 1:
            dx = 0.4
        ax.arrow(x+0.5, 0.5, dx, dy, head_width=0.1, head_length=0.1, fc='black')
    plt.show()

def plot_secret_env(V):
    matrix = [[V.get(y) for y in range(len(V))] for x in range(1)]

    fig, ax = plt.subplots(figsize=(8, 1))
    sns.heatmap(matrix, cmap='coolwarm', cbar=True, ax=ax)

    ax.set_xlim(0, len(V))
    ax.set_ylim(0, 1)
    plt.show()

def policy_evaluation_on_line_world() -> ValueFunction:
    """
    Creates a Line World of 7 cells (leftmost and rightmost are terminal, with -1 and 1 reward respectively)
    Launches a Policy Evaluation Algorithm in order to find the Value Function of a uniform random policy
    Returns the Value function (V(s)) of this policy
    """
    S, A, R = env_line_world()
    return policy_evaluation(S, A, R, p_line_world, pi_random_line_world)


def policy_iteration_on_line_world() -> PolicyAndValueFunction:
    """
    Creates a Line World of 7 cells (leftmost and rightmost are terminal, with -1 and 1 reward respectively)
    Launches a Policy Iteration Algorithm in order to find the Optimal Policy and its Value Function
    Returns the Policy (Pi(s,a)) and its Value Function (V(s))
    """
    S, A, R = env_line_world()
    pi, V = policy_iteration(S, A, R, p_line_world)
    plot_line_world(pi, V)
    return pi, V


def value_iteration_on_line_world() -> PolicyAndValueFunction:
    """
    Creates a Line World of 7 cells (leftmost and rightmost are terminal, with -1 and 1 reward respectively)
    Launches a Value Iteration Algorithm in order to find the Optimal Policy and its Value Function
    Returns the Policy (Pi(s,a)) and its Value Function (V(s))
    """
    S, A, R = env_line_world()
    pi, V = value_iteration(S, A, R, p_line_world)
    plot_line_world(pi, V)
    return pi, V



def policy_evaluation_on_grid_world() -> ValueFunction:
    """
    Creates a Grid World of 5x5 cells (upper rightmost and lower rightmost are terminal, with -1 and 1 reward respectively)
    Launches a Policy Evaluation Algorithm in order to find the Value Function of a uniform random policy
    Returns the Value function (V(s)) of this policy
    """
    S, A, R = env_grid_world()
    V = policy_evaluation(S, A, R, p_grid_world, pi_random_grid_world)
    plot_grid_world(None, V)
    return V


def policy_iteration_on_grid_world() -> PolicyAndValueFunction:
    """
    Creates a Grid World of 5x5 cells (upper rightmost and lower rightmost are terminal, with -1 and 1 reward respectively)
    Launches a Policy Iteration Algorithm in order to find the Optimal Policy and its Value Function
    Returns the Policy (Pi(s,a)) and its Value Function (V(s))
    """
    S, A, R = env_grid_world()
    pi, V = policy_iteration(S, A, R, p_grid_world)
    plot_grid_world(pi, V)
    return pi, V



def value_iteration_on_grid_world() -> PolicyAndValueFunction:
    """
    Creates a Grid World of 5x5 cells (upper rightmost and lower rightmost are terminal, with -1 and 1 reward respectively)
    Launches a Value Iteration Algorithm in order to find the Optimal Policy and its Value Function
    Returns the Policy (Pi(s,a)) and its Value Function (V(s))
    """
    S, A, R = env_grid_world()
    pi, V = value_iteration(S, A, R, p_grid_world)
    plot_grid_world(pi, V)
    return pi, V


def policy_evaluation_on_secret_env1() -> ValueFunction:
    """
    Creates a Secret Env1
    Launches a Policy Evaluation Algorithm in order to find the Value Function of a uniform random policy
    Returns the Value function (V(s)) of this policy
    """
    env = Env1()
    # TODO
    S = env.states()
    A = env.actions()
    R = env.rewards()

    def pi_random(s, a):
        return 1 / len(A)

    return policy_evaluation(S, A, R, env.transition_probability, pi_random)



def policy_iteration_on_secret_env1() -> PolicyAndValueFunction:
    """
    Creates a Secret Env1
    Launches a Policy Iteration Algorithm in order to find the Optimal Policy and its Value Function
    Returns the Policy (Pi(s,a)) and its Value Function (V(s))
    """
    env = Env1()
    # TODO
    S = env.states()
    A = env.actions()
    R = env.rewards()

    return policy_iteration(S, A, R, env.transition_probability)


def value_iteration_on_secret_env1() -> PolicyAndValueFunction:
    """
    Creates a Secret Env1
    Launches a Value Iteration Algorithm in order to find the Optimal Policy and its Value Function
    Prints the Policy (Pi(s,a)) and its Value Function (V(s))
    """
    env = Env1()
    # TODO
    S = env.states()
    A = env.actions()
    R = env.rewards()

    pi, V = value_iteration(S, A, R, env.transition_probability)

    plot_secret_env(V)

    return pi, V


def demo():
    pass
    # print(policy_evaluation_on_line_world())
    # print(policy_iteration_on_line_world())
    # print(value_iteration_on_line_world())
    #
    # print(policy_evaluation_on_grid_world())
    # print(policy_iteration_on_grid_world())
    print(value_iteration_on_grid_world())
    #
    # print(policy_evaluation_on_secret_env1())
    # print(policy_iteration_on_secret_env1())
    # print(value_iteration_on_secret_env1())
