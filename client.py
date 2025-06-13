#Aqui vocês irão colocar seu algoritmo de aprendizado
import connection as cn
import numpy as np

# def choose_best_action(q_matrix, state):
#     """
#     Função que retorna o índice da melhor ação (com maior valor Q)
#     para o estado atual.
#     """
#     act = np.argmax(q_matrix[state])  # retorna índice (0-based)
#     return act

# Função para executar a política aprendida e registrar a recompensa total acumulada

# def simulate_policy(q_matrix, rw):

#   r_total = 0

#   state = 0  # estado inicial

#   terminal = True

#   while terminal:

#       # Escolher ação com base na política aprendida
#       action_trial = choose_best_action(q_matrix, state)

#       # Selecionar a matriz de transição correspondente à ação
#       if action_trial == 0:
#           transition_state = T_up[state]
#       elif action_trial == 1:
#           transition_state = T_down[state]
#       elif action_trial == 2:
#           transition_state = T_left[state]
#       elif action_trial == 3:
#           transition_state = T_right[state]

#       # Aplicar a ação e observar o próximo estado
#       next_state = calc_action_result(state, transition_state)

#       print(f"{state} {actions_names[action_trial]} {next_state}")

#       # Acumular recompensa
#       r_total += rw[next_state]

#       # Atualizar estado
#       state = next_state

#       # Verificar se é estado terminal
#       if state == 9 or state == 10:  # estados terminais 9 e 10 em Python
#         terminal = False

#   # Resultado total acumulado
#   return r_total

s = cn.connect(2037)



N_matrix = np.zeros((24, 3))  # Matriz de contagem de ações
actions = ["left", "right", "jump"]

T = 5

for i in range(T):

    print(f"Trajetoria {i}")
    state = 0
    terminal = True
    while terminal:
        action = actions[np.random.choice([0, 1, 2])]  # Escolhe uma ação aleatória
        N_matrix[state, action] += 1

        # Envia a ação e recebe o novo estado e recompensa
        next_state, reward = cn.get_state_reward(s, "right")
        print(f"Ação: {action}, Estado: {next_state}, Recompensa: {reward}")
        facing = next_state[-2:]
        next_state = int(next_state[2:7],2)
      

# state, reward = cn.get_state_reward(s, "jump")
# print(f"Ação: {actions[0]}, Estado: {state}, Recompensa: {reward}")
