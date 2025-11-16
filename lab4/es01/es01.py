from qubovert import boolean_var
from qubovert import QUBO
from qubovert.sim import anneal_qubo
from typing import Dict, List, Tuple

def read_graph(path: str) -> Tuple[int, List[Tuple[int, int, float]]]:

    edges: List[Tuple[int, int, float]] = []
    with open(path, "r") as f:
        lines = [ln.strip() for ln in f if ln.strip() and not ln.strip().startswith("#")]
    if not lines:
        raise ValueError("Empty file or only comments.")
    n = int(lines[0])
    for ln in lines[1:]:
        parts = ln.split()
        if len(parts) != 3:
            raise ValueError(f"Bad edge line: {ln!r}")
        i, j = int(parts[0]), int(parts[1])
        w = float(parts[2])
        if i == j:
            # self-loops do not contribute to cut; ignore or raise
            continue
        if j < i:
            i, j = j, i
        edges.append((i, j, w))

    # basic validation
    for (i, j, _) in edges:
        if not (0 <= i < n and 0 <= j < n):
            raise ValueError(f"Edge ({i},{j}) has node index outside 0..{n-1}")

    return n, edges

def build_maxcut_qubo(n: int, edges: List[Tuple[int, int, float]]) -> Tuple[QUBO, Dict[int, object]]:

    x = {i: boolean_var(f"x{i}") for i in range(n)}
    H = QUBO()
    for i, j, w in edges:
        H += w * (2 * x[i] * x[j] - x[i] - x[j])
    return H, x

def solve_and_verify(H: QUBO, edges: List[Tuple[int, int, float]]) -> Tuple[Dict[str, int], float, float]:

    sol = anneal_qubo(H)              

    return sol

def main(path="Lab4/es01/max_cut.txt"):
    n, edges = read_graph(path)
    H, x = build_maxcut_qubo(n, edges)
    res = anneal_qubo(H)
    
    for r in res:
        print(r.state, "-> energia:", r.value)


if __name__ == "__main__":
    main()
