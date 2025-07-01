import connection as cn
import numpy as np

s = cn.connect(2037) # Conexão com o servidor

q_matrix = np.zeros((96, 3))  # Matriz Q inicializada contagem com zeros
N_matrix = np.zeros((96, 3))  # Matriz de contagem de ações
actions = ["left", "right", "jump"] # Lista de ações possíveis

alpha = 0.3 # taxa de aprendizado
gamma = 0.95 # fator de desconto

T = 500  # Número de trajetórias a serem executadas

def q_update(state, action, next_state, reward, q_matrix, alpha=0.3, gamma=0.95): # Q-learning (https://en.wikipedia.org/wiki/Q-learning)
    estimate_q = reward + gamma * np.max(q_matrix[next_state, :])
    q_value = q_matrix[state, action] + alpha * (estimate_q - q_matrix[state, action])
    return q_value

def choose_action(state, q_matrix, N_matrix, c=2.0): # usando UCB (https://banditalgs.com/2016/09/18/the-upper-confidence-bound-algorithm/)
    total_visits = np.sum(N_matrix[state]) + 1e-5  # evitando log(0)
    q_values = q_matrix[state]
    counts = N_matrix[state] + 1e-5  # evitando divisão por zero
    log_total = np.log(max(total_visits, 1.0001))  # garantindo log positivo
    ucb_bonus = c * np.sqrt(log_total / counts)
    ucb_scores = q_values + ucb_bonus
    ucb_scores = np.nan_to_num(ucb_scores, nan=-np.inf)  # anulando NaNs
    return actions[np.argmax(ucb_scores)]

def print_result(q_matrix, file): # função para escrever os arquivos
    with open(file, "w") as f:
        for row in q_matrix:
            f.write(f" ".join(map(lambda x: f"{x:.6f}", row)) + "\n")

mode = input("1: Treinamento, 2: Teste\n")

if(int(mode) == 1):
    sucesso = 0
    for i in range(T):
        print(f"Trajetoria {i} - Sucessos: {sucesso}")
        state = 0
        terminal = True
        while terminal:
            action =  choose_action(state, q_matrix, N_matrix, c=3.0) # Escolhe uma ação aleatória
            action_index = actions.index(action)
            N_matrix[state, action_index] += 1  # Incrementa a contagem de ações
            next_state, reward = cn.get_state_reward(s, action) # Envia a ação e recebe o novo estado e recompensa
            next_state = int(next_state[2:], 2)

            # atualização da tabela de recompensas por estado
            q_matrix[state, action_index] = q_update(state, action_index, next_state, reward, q_matrix, alpha=alpha, gamma=gamma)
            
            state = next_state

            if reward == 300:
                sucesso += 1
                terminal = False
            if reward == -100:
                terminal = False
    # criar arquivos de resultados após os loops
    print_result(q_matrix, "resultado.txt")
    print_result(N_matrix, "actions.txt")
    
elif (int(mode) == 2):
    q_matrix = np.loadtxt("resultado.txt", delimiter=" ")
    terminal = True
    state = 0
    while terminal:
        action = actions[np.argmax(q_matrix[state, :])]
        next_state, reward = cn.get_state_reward(s, action)
        platform = int(next_state[2:7], 2)
        direction = int(next_state[7:], 2)
        cardinal_directions = ["Norte", "Leste", "Sul", "Oeste"]
        next_state = int(next_state[2:], 2)
        print(f"Ação: {action}, Plataforma: {platform}, Direção: {cardinal_directions[direction]}, Recompensa: {reward}")
        state = next_state
        if reward == 300:
            input("Success. Press enter to leave.")
            terminal = False
        if reward == -100:
            input("Failure. Press enter to leave.")
            terminal = False
else:
    input("Unexpected input. Press enter to leave.")