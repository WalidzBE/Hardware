from qubovert import QUBO, boolean_var
from qubovert.sim import anneal_qubo
import math

def count_unsatisfied_clauses(clauses, x):
    def lit_value(l):
        val = x[abs(l)-1]   # x array is 0-indexed, literals are 1-indexed
        return val if l > 0 else (1 - val)

    unsatisfied = []
    for idx, clause in enumerate(clauses, start=1):
        satisfied = any(lit_value(l) == 1 for l in clause)
        if not satisfied:
            unsatisfied.append(idx)

    return len(unsatisfied), unsatisfied

def decode_state(state, n_vars):
    print(len(state))
    selected = [state.get(i, 0) for i in range(n_vars)]

    print("Possible Solution:", selected)
    print()
    print("Clauses:")

    unsat_count, unsat_list = count_unsatisfied_clauses(clauses, selected)

    print("Unsatisfied clauses:", unsat_list)
    print("Number of unsatisfied:", unsat_count)

def literal_expr(x_vars, l):
    if l > 0:
        return x_vars[l - 1]
    else:
        return 1 - x_vars[abs(l) - 1]

def build_qubo_from_clauses(clauses, n_vars, alpha=20, lambda_penalty=5):
    # Base decision variables
    x = [boolean_var(f"x{i+1}") for i in range(n_vars)]

    H_total = 0

    # Iterate over clauses
    for c_idx, clause in enumerate(clauses):
        l1, l2, l3 = [literal_expr(x, l) for l in clause]

        y = boolean_var(f"y{c_idx+1}")

        # Clause penalty (unsatisfied)
        clause_penalty = 1 - (l1 + l2 + l3) + (l1*l2 + l1*l3 + l2*l3) - y*l3
        # Constraint penalty enforcing y = l1*l2
        constraint_penalty = l1*l2 - 2*l1*y - 2*l2*y + 3*y

        # Combine
        H_total += alpha * clause_penalty + lambda_penalty * constraint_penalty

    # Convert to QUBO and solve
    qubo = H_total.to_qubo()
    anneal_qubo_results= anneal_qubo(qubo, num_anneals=500)

    # Sort by energy (value)
    best_state = min(anneal_qubo_results, key=lambda s: s.value)
    return best_state

filename = "lab4/files/clauses.txt"

# Read file
with open(filename, "r") as f:
    lines = [line.strip() for line in f if line.strip()]
n_vars, n_clauses = map(int, lines[0].split())
clauses = [list(map(int, line.split())) for line in lines[1:]]

# Build QUBO
lambda_penalty = 5
alpha = 20
results = build_qubo_from_clauses(clauses,n_vars,alpha,lambda_penalty)

print(results)
decode_state(results.state,n_vars)
