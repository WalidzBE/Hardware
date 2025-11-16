import ast
import re

import matplotlib.pyplot as plt
import networkx as nx
from qubovert import QUBO, boolean_var
from qubovert.sim import anneal_qubo


def read_graph_coloring_file(filename):
    edges = []

    with open(filename, "r") as f:
        first_line = f.readline().strip().split()
        num_nodes = int(first_line[0])  # read numbers of nodes
        num_colors = int(first_line[1])  # read numbers of colours

        for line in f:  # read edges
            if line.strip():
                i, j = map(int, line.split())
                edges.append((i, j))

    return num_nodes, num_colors, edges


# **********************************
# MAIN
# **********************
filename = "graph_coloring.txt"
num_nodes, num_colors, edges = read_graph_coloring_file(filename)

print("Numero nodi:", num_nodes)
print("Numero colori:", num_colors)
print("Archi:", edges)


# 2. Crea le variabili x_{i,c}

# each node i we assign c --> 0 or 1
x = {
    (i, c): boolean_var(f"x_{i}_{c}")  # create QUBO variables
    for i in range(num_nodes)  # node i
    for c in range(num_colors)
}  # colour c

#  3. Inizializza il QUBO
model = QUBO()

# 4. Vincolo: ogni nodo ha un solo colore

# square penaly is zero only if exactly one color is chosen
for i in range(num_nodes):
    model += (1 - sum(x[(i, c)] for c in range(num_colors))) ** 2


# 5. Vincolo: nodi adiacenti non devono avere lo stesso colore

# if both nodes of an edge have same colour --> penalty added
for i, j in edges:
    for c in range(num_colors):
        model += x[(i, c)] * x[(j, c)]


# 6. Risolvi con simulated annealing
result = anneal_qubo(model)
print(result)

# *********************************************
# ESTRAZIONE DELLA SOLUZIONE DAL PRINT
# *************************

result_str = str(result)  # converti tutto in stringa

# Cerca il dizionario dopo "state:"
match = re.search(r"state:\s*(\{.*\})", result_str)
state_str = match.group(1)  # prende la parte {...}

# Converti la stringa in dizionario
solution = ast.literal_eval(state_str)

# *****************************************
# COSTRUZIONE COLORAZIONE
# *************************

coloring = {}

for i in range(num_nodes):
    for c in range(num_colors):
        var_name = f"x_{i}_{c}"
        if solution[var_name] == 1:  # x_0_1 : 1
            coloring[i] = c
            break

print("\nColorazione finale:")
for node in range(num_nodes):
    print(f"Nodo {node} → Colore {coloring[node]}")


# *************************
# GRAFICO
# *************************

G = nx.Graph()
G.add_edges_from(edges)

node_colors = [coloring[i] for i in G.nodes()]

plt.figure(figsize=(6, 6))
nx.draw(
    G,
    pos=nx.spring_layout(G, seed=42),
    with_labels=True,
    node_color=node_colors,
    cmap="Pastel1",
    node_size=950,
    font_size=16,
    font_weight="bold",
    edge_color="#555555",  # grigio scuro
)

plt.title("Colorazione del grafo (Simulated Annealing)", fontsize=18, fontweight="bold")
plt.show()
