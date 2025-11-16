
import numpy as np
import pandas as pd
from qubovert import boolean_var
from qubovert.sim import anneal_qubo


def build_qubo_from_affinity(A: np.ndarray, T: int, S: int, penalty_scale: float = 40.0):

    N = A.shape[0]
    assert N == T * S, "N must equal T*S"

    # decision variables
    xobjs  = {(i, t): boolean_var(f"x_{i}_{t}") for i in range(N) for t in range(T)}
    xnames = {(i, t): f"x_{i}_{t}" for i in range(N) for t in range(T)}

    row_sum_abs = np.sum(np.abs(A), axis=1)
    M = float(np.max(row_sum_abs))
    scale = penalty_scale * (M + 1.0)
    A_pen = scale
    B_pen = scale

    H = 0

    # objective: - sum_{i<j} a_ij * sum_t x_{i,t} x_{j,t}
    for i in range(N):
        for j in range(i+1, N):
            aij = float(A[i, j])
            if aij == 0.0:
                continue
            for t in range(T):
                H += -aij * (xobjs[(i, t)] * xobjs[(j, t)])

    # guest constraint
    for i in range(N):
        s = 0
        for t in range(T):
            s += xobjs[(i, t)]
        H += A_pen * (1 - s) ** 2

    # table constraint
    for t in range(T):
        s = 0
        for i in range(N):
            s += xobjs[(i, t)]
        H += B_pen * (S - s) ** 2

    return H, xobjs, xnames, A_pen, B_pen


def scores(A: np.ndarray, assign: list):
    N = A.shape[0]
    raw_same = 0.0
    pos_together = 0.0
    neg_sep_reward = 0.0
    for i in range(N):
        for j in range(i+1, N):
            same = int(assign[i] == assign[j])
            aij = float(A[i, j])
            raw_same += aij * same
            if aij > 0:
                pos_together += aij * same
            elif aij < 0:
                neg_sep_reward += (-aij) * (1 - same)
    return raw_same, pos_together, neg_sep_reward




if __name__ == "__main__":
    # problem params
    T = 4
    S = 5
    PENALTY_SCALE = 40.0
    NUM_ANNEALS = 12000

    # load guests.csv
    df = pd.read_csv("guests.csv", header=0, index_col=0).fillna(0.0)
    A = df.values.astype(float)
    N = A.shape[0]
    for i in range(N):
        A[i, i] = 0.0
    A = (A + A.T) / 2.0

    # build QUBO expression
    H_expr, xobjs, xnames, A_pen, B_pen = build_qubo_from_affinity(A, T, S, penalty_scale=PENALTY_SCALE)

    varorder = [f"x_{i}_{t}" for i in range(N) for t in range(T)]

    # Convert to a QUBO dict
    Q = H_expr.to_qubo()  

    res = anneal_qubo(Q, NUM_ANNEALS)

    best = min(res, key=lambda s: s.value)
    print("Best annealed energy (includes penalties):", best.value)

    state = best.state  
    print(state)
