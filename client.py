import connection as cn
import numpy as np

# Conexão com o servidor
s = cn.connect(2037)

q_matrix = np.zeros((96, 3))  # Matriz Q inicializada com zeros
# 96 estados (0 a 95) e 3 ações (left, right, jump)
# A matriz Q terá dimensões (número de estados, número de ações)

N_matrix = np.zeros((96, 3))  # Matriz de contagem de ações
actions = ["left", "right", "jump"] # Lista de ações possíveis

alpha = 0.1  # Taxa de aprendizado (0.2 é o valor do exemplo no notebook)
gamma = 0.9  # Fator de desconto (0.5 é o valor do exemplo no notebook)

T = 2500  # Número de trajetórias a serem executadas

def q_update(state, action, next_state, reward, q_matrix, alpha, gamma):
    estimate_q = reward + gamma * np.max(q_matrix[next_state, :])
    q_value = q_matrix[state,action] + alpha * (estimate_q - q_matrix[state][action])
    return q_value

for i in range(T):

    print(f"Trajetoria {i}")
    terminal = True
    state = 0
    while terminal:
        action = actions[np.random.choice([0, 1, 2])]  # Escolhe uma ação aleatória

        N_matrix[state][actions.index(action)] += 1  # Incrementa a contagem de ações

        # Envia a ação e recebe o novo estado e recompensa
        next_state, reward = cn.get_state_reward(s, action)
        next_state = int(next_state[2:], 2)

        q_matrix[next_state][actions.index(action)] = q_update(state, actions.index(action), next_state, reward, q_matrix, alpha, gamma)
        print(f"Ação: {action}, Estado: {next_state}, Recompensa: {reward}")
        
        state = next_state

        if reward == 300 or reward == -100:
            terminal = False

# criar arquivo de resultados
with open("result.txt", "w") as f:
    for row in q_matrix:
        f.write(f" ".join(map(lambda x: f"{x:.6f}", row)) + "\n")

# for i in range (T):
#     state, reward = cn.get_state_reward(s, "jump")
#     next_state = int(state[2:],2)
#     print(f"Ação: {actions[0]}, Estado: {next_state}, Recompensa: {reward}")