import subprocess
import itertools
import csv
import os

# Parâmetros a testar
alphas = [0.1, 0.3, 0.5]
gammas = [0.85, 0.95, 0.99]
e_decays = [0.99, 0.995, 0.999]

# Resultado: lista de dicionários
results = []

# Número de trajetórias por teste (deixe pequeno no início, ex: 500)
T = 500

# Garante que os testes rodam com os mesmos parâmetros iniciais
os.environ["PYTHONUNBUFFERED"] = "1"  # Força print imediato no subprocess

for alpha, gamma, e_decay in itertools.product(alphas, gammas, e_decays):
    print(f"Testando: alpha={alpha}, gamma={gamma}, e_decay={e_decay}")
    
    # Executa client.py com parâmetros passados via linha de comando
    process = subprocess.Popen(
        ["python", "client.py", "--mode", "1", "--alpha", str(alpha),
         "--gamma", str(gamma), "--e_decay", str(e_decay), "--T", str(T)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True
    )

    sucesso = 0
    for line in process.stdout:
        print(line.strip())
        if "Sucesso" in line:
            try:
                sucesso = int(line.strip().split("Sucesso")[-1])
            except:
                pass

    process.wait()
    results.append({
        "alpha": alpha,
        "gamma": gamma,
        "e_decay": e_decay,
        "sucessos": sucesso,
        "sucesso_percent": round(sucesso / T * 100, 2)
    })

# Salva como CSV
with open("qlearning_results.csv", "w", newline="") as csvfile:
    fieldnames = ["alpha", "gamma", "e_decay", "sucessos", "sucesso_percent"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for r in results:
        writer.writerow(r)

print("Testes finalizados. Resultados em qlearning_results.csv")