from qubovert import QUBO, boolean_var
from qubovert.sim import anneal_qubo

# 1. Leggi il file graph_coloring.txt
# -> estrai numero_nodi, numero_colori, lista degli archi

# 2. Crea le variabili x_{i,c}
x = {
    (i, c): boolean_var(f"x_{i}_{c}")
    for i in range(num_nodi)
    for c in range(num_colori)
}

# 3. Inizializza il QUBO
model = QUBO()

# 4. Vincolo: ogni nodo ha un solo colore
for i in range(num_nodi):
    model += (1 - sum(x[(i, c)] for c in range(num_colori))) ** 2

# 5. Vincolo: nodi adiacenti non devono avere lo stesso colore
for i, j in edges:
    for c in range(num_colori):
        model += x[(i, c)] * x[(j, c)]

# 6. Risolvi con simulated annealing
solution = anneal_qubo(model)

print(solution)
