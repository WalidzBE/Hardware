from qubovert import QUBO, boolean_var
from qubovert.sim import anneal_qubo
import math

def decode_state(state, objects, Wmax):
    n_objects = len(objects)
    num_s_bits = len(state) - n_objects
    selected = [i for i in range(n_objects) if state.get(i, 0) == 1]
    total_pref = sum(objects[i][0] for i in selected)
    total_weight = sum(objects[i][1] for i in selected)
    slack = sum((2 ** j) * state.get(n_objects + j, 0) for j in range(num_s_bits))
    return selected, total_pref, total_weight, slack

def qubo_formulation_luggage_problem(num_objects, max_weight,objects):

    # Define binary decision vars
    x = [boolean_var(f"x{i}") for i in range(len(objects))]

    # Define slack variable (binarized)
    num_s_bits = math.ceil(math.log2(max_weight + 1))
    s = [boolean_var(f"s{i}") for i in range(num_s_bits)]
    s_value = sum((2 ** i) * s[i] for i in range(num_s_bits))

    # Objective: maximize preferences
    # QUBO minimizes, so we use the negative
    P = sum(-p * x[i] for i, (p, _) in enumerate(objects))

    # Constraint penalty
    λ = 10 * max(p for p, _ in objects)
    weight_expr = sum(w * x[i] for i, (_, w) in enumerate(objects)) + s_value - max_weight
    constraint_penalty = λ * (weight_expr ** 2)

    # Combine into total Hamiltonian
    H = P + constraint_penalty

    # Convert to QUBO and solve
    qubo = H.to_qubo()
    anneal_qubo_results= anneal_qubo(qubo, num_anneals=500)

    # Sort by energy (value)
    best_state = min(anneal_qubo_results, key=lambda s: s.value)
    return best_state

# Start of the main code
file_path = "lab4/files/objects.txt"

# Read data from file
with open(file_path, 'r') as file:
    data = file.read().splitlines()

num_objects, max_weight = map(int, data[0].split())

# The rest: preference and weight pairs
objects = []
for item in data[1:]:
    preference, weight = map(int, item.split())
    objects.append((preference, weight))

print(f"Number of objects: {num_objects}")
print(f"Maximum weight: {max_weight} kg")
print(f"Objects: {objects}")
results = qubo_formulation_luggage_problem(num_objects, max_weight, objects)


selected, total_pref, total_weight, slack = decode_state(results.state, objects, max_weight)
print("Selected objects:", selected)
print(f"Total weight: {total_weight:.2f} kg (limit: {max_weight})")
print(f"Total preference score: {total_pref:.2f}")
