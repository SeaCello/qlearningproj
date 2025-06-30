import subprocess
import itertools
import os
import argparse
import csv
import math
from collections import deque

# ================================
# Argumentos de linha de comando
# ================================
parser = argparse.ArgumentParser()
parser.add_argument("--mode", type=int, default=1, help="1: Treinamento, 2: Teste")
parser.add_argument("--alpha", type=float, default=0.9)
parser.add_argument("--gamma", type=float, default=0.95)
parser.add_argument("--epsilon", type=float, default=1.0)
parser.add_argument("--e_decay", type=float, default=0.9995)
parser.add_argument("--e_min", type=float, default=0.01)
parser.add_argument("--T", type=int, default=10000)
parser.add_argument("--policy", type=str, default="greedy", choices=["greedy", "softmax", "ucb"])
parser.add_argument("--tau", type=float, default=1.0, help="Temperatura para softmax")
parser.add_argument("--ucb_c", type=float, default=2.0, help="Constante de confiança para UCB")


args = parser.parse_args()

# ================================
# Inicialização
# ================================
s = cn.connect(2037)

q_matrix = np.zeros((96, 3))
N_matrix = np.zeros((96, 3))
actions = ["left", "right", "jump"]

alpha = args.alpha
gamma = args.gamma
epsilon = args.epsilon
e_decay = args.e_decay
e_min = args.e_min
T = args.T

def q_update(state, action, next_state, reward, q_matrix, alpha, gamma):
    estimate_q = reward + gamma * np.max(q_matrix[next_state, :])
    q_value = q_matrix[state, action] + alpha * (estimate_q - q_matrix[state, action])
    return q_value

def choose_action(state, epsilon):
    if np.random.uniform(0, 1) < epsilon:
        return actions[np.random.choice([0, 1, 2])]
    else:
        return actions[np.argmax(q_matrix[state, :])]

def greedy_action(state, epsilon):
    if np.random.uniform(0, 1) < epsilon:
        return actions[np.random.choice([0, 1, 2])]
    return actions[np.argmax(q_matrix[state])]

def softmax_action(state, tau):
    q_vals = q_matrix[state]
    max_q = np.max(q_vals)  # estabilidade numérica
    exp_q = np.exp((q_vals - max_q) / tau)
    probs = exp_q / np.sum(exp_q)
    return np.random.choice(actions, p=probs)

def ucb_action(state, c):
    visits = N_matrix[state] + 1e-5
    total = np.sum(visits)
    log_total = math.log(max(total, 1.0001))
    bonuses = c * np.sqrt(log_total / visits)
    scores = q_matrix[state] + bonuses
    scores = np.nan_to_num(scores, nan=-np.inf)
    return actions[np.argmax(scores)]

def choose_action(state, epsilon, policy):
    if policy == "greedy":
        return greedy_action(state, epsilon)
    elif policy == "softmax":
        return softmax_action(state, args.tau)
    elif policy == "ucb":
        return ucb_action(state, args.ucb_c)
    else:
        raise ValueError("Política de ação inválida")


# ================================
# Modo Treinamento
# ================================
if args.mode == 1:
    sucesso = 0
    for i in range(T):
        print(f"Trajetoria {i}, Epsilon {epsilon:.4f}, Sucesso {sucesso}")
        state = 0
        terminal = True
        while terminal:
            action = choose_action(state, epsilon)
            action_index = actions.index(action)
            N_matrix[state][action_index] += 1

            next_state, reward = cn.get_state_reward(s, action)
            next_state = int(next_state[2:], 2)

            q_matrix[state][action_index] = q_update(state, action_index, next_state, reward, q_matrix, alpha, gamma)

            if reward == 300:
                sucesso += 1
                terminal = False
            elif reward == -100:
                terminal = False

            state = next_state

        epsilon = max(e_min, epsilon * e_decay)

    # Salva Q-table
    with open("result.txt", "w") as f:
        for row in q_matrix:
            f.write(" ".join(f"{x:.6f}" for x in row) + "\n")

    with open("actions.txt", "w") as f:
        for row in N_matrix:
            f.write(" ".join(str(int(x)) for x in row) + "\n")

# ================================
# Modo Teste
# ================================
elif args.mode == 2:
    q_matrix = np.loadtxt("result.txt", delimiter=" ")
    state = 0
    terminal = True
    while terminal:
        action = actions[np.argmax(q_matrix[state, :])]
        next_state, reward = cn.get_state_reward(s, action)
        next_state = int(next_state[2:], 2)
        print(f"Ação: {action}, Estado: {next_state}, Recompensa: {reward}")
        if reward == 300 or reward == -100:
            terminal = False
        state = next_state
else:
    print("Modo inválido. Use --mode 1 para treinamento ou --mode 2 para teste.")
