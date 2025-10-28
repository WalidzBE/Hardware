# compiler_basic.py
# Basic, non-optimizing compiler for FakeGuadalupeV2:
# - inline composite gates; lower mcx
# - translate to native {rz, sx, x} (1q) + {cx} (2q)
# - trivial mapping: logical i -> physical i
# - naive SWAP routing on the coupling map with orientation handling
# - **emit directly on logical wires** (no final relabeling step)

from collections import deque
from typing import Dict, List, Tuple

from qiskit import QuantumCircuit
from qiskit.circuit import ControlledGate
from qiskit.circuit.library import UGate
from qiskit.quantum_info import Operator

# OneQubitEulerDecomposer import (version-tolerant)
try:
    from qiskit.synthesis import OneQubitEulerDecomposer
except Exception:
    try:
        from qiskit.quantum_info.synthesis import OneQubitEulerDecomposer
    except Exception:
        from qiskit.synthesis.one_qubit.euler_decomposer import OneQubitEulerDecomposer

# MCX gate (we'll decompose and inline it)
try:
    from qiskit.circuit.library.standard_gates import MCXGate
except Exception:
    MCXGate = None

# Coupling map import (version-tolerant)
try:
    # Qiskit 0.46+ backend API
    from qiskit.transpiler.coupling import CouplingMap
except Exception:
    from qiskit.transpiler import CouplingMap


# ---------- backend topology helpers ----------

def _build_coupling_map(backend) -> CouplingMap:
    """Get a CouplingMap from a BackendV2 (FakeGuadalupeV2)."""
    tgt = getattr(backend, "target", None)
    if tgt is not None and hasattr(tgt, "build_coupling_map"):
        return tgt.build_coupling_map()
    if hasattr(backend, "coupling_map") and backend.coupling_map is not None:
        return backend.coupling_map
    cfg = getattr(backend, "configuration", lambda: None)()
    if cfg is not None and getattr(cfg, "coupling_map", None):
        return CouplingMap(couplinglist=cfg.coupling_map)
    raise RuntimeError("Cannot obtain a coupling map from backend.")


def _directed_edges(cmap: CouplingMap) -> set:
    try:
        return set(map(tuple, cmap.get_edges()))
    except Exception:
        return set(tuple(e) for e in cmap)


def _adjacency_undirected(cmap: CouplingMap, n_qubits: int) -> List[set]:
    """Undirected adjacency for BFS path finding."""
    adj = [set() for _ in range(n_qubits)]
    for u, v in _directed_edges(cmap):
        adj[u].add(v)
        adj[v].add(u)
    return adj


# ---------- compiler ----------

class BasicGuadalupeCompiler:
    """
    Simple compiler:
      - Pre-pass: inline composite gates recursively; lower MCX via Qiskit; rely on .definition for CCX/CCZ
      - Basis translation to {rz, sx, x} (1q) and {cx} (2q)
      - Trivial layout + SWAP routing (shortest-path BFS) on the coupling map
      - Orientation handled by local H wrappers
      - **Emit on logical wires as we go; no final wire relabeling**
    """

    def __init__(self, backend, restore_logical_order: bool = True):
        self.backend = backend
        self.n_phys = getattr(backend, "num_qubits", None) or backend.configuration().n_qubits
        self.cmap = _build_coupling_map(backend)
        self.dir_edges = _directed_edges(self.cmap)                  # set of (u,v)
        self.adj = _adjacency_undirected(self.cmap, self.n_phys)     # undirected for routing
        self.decomposer = OneQubitEulerDecomposer(basis="ZSX")       # -> rz, sx, rz, sx, rz
        self.native_1q = {"rz", "sx", "x", "id"}
        self.restore_logical_order = restore_logical_order

    # ---------- Public API ----------

    def compile(self, circ: QuantumCircuit) -> Tuple[QuantumCircuit, Dict[int, int], Dict[int, int]]:
        """
        Compile for FakeGuadalupeV2:
        1) strip final measurements/barriers
        2) inline composites
        3) translate to {rz, sx, x, cx}
        4) route cx by inserting LOGICAL swaps; keep a running wire permutation pi
            so *all subsequent gates* are emitted on the right wires.
        Returns (compiled_circuit, final_layout (logical->physical), pi (logical->wire)).
        """
        n_log = circ.num_qubits
        if n_log > self.n_phys:
            raise ValueError(f"Circuit uses {n_log} qubits but backend has only {self.n_phys}.")

        unitary    = self._strip_final_measurements(circ)
        flattened  = self._inline_composites(unitary)
        translated = self._translate_to_native(flattened)

        # Routing placement used to *decide* paths (logical -> physical)
        layout: Dict[int, int] = {l: l for l in range(n_log)}
        # Output wire permutation: pi[l] = current output wire for logical l
        pi: Dict[int, int] = {l: l for l in range(n_log)}

        out = QuantumCircuit(n_log)

        for ci in translated.data:
            inst = getattr(ci, "operation", getattr(ci, "instruction", None)) or ci[0]
            qargs = getattr(ci, "qubits", None)
            if qargs is None:
                qargs = ci[1]

            if len(qargs) == 1:
                l = qargs[0]._index
                w = pi[l]  # <- emit on current wire for this logical
                if inst.name in self.native_1q:
                    self._emit_1q(out, inst.name, getattr(inst, "params", []), w)
                else:
                    # safety fallback; shouldn't occur post-translation
                    from qiskit.quantum_info import Operator
                    U = Operator(inst).data
                    for op, pars in self._decompose_1q_to_zsx(U):
                        self._emit_1q(out, op, pars, w)

            elif len(qargs) == 2:
                lc = qargs[0]._index
                lt = qargs[1]._index
                # Route and EMIT logical swaps; update layout and pi together; emit final CX on wires (pi[lc], pi[lt])
                self._route_and_emit_cx_LOGICAL(out, lc, lt, layout, pi)
            else:
                raise NotImplementedError(f"Unexpected multi-qubit instruction: {inst.name}")

        if self.restore_logical_order:
            # Emit logical swaps to make layout[l] == l, keeping pi updated as we swap
            self._restore_order_LOGICAL(out, layout, pi, n_log)

        return out, layout, pi


    # ---------- Composite inlining & lowering ----------

    def _inline_composites(self, circ: QuantumCircuit) -> QuantumCircuit:
        """Recursively inline any instruction that has a .definition into only primitive ops."""
        out = QuantumCircuit(circ.num_qubits)
        changed = False
        for inst, qargs, cargs in circ.data:
            name = inst.name
            if name in {"barrier", "reset", "measure"}:
                out.append(inst, qargs, cargs)
                continue

            # Use .definition when available (covers CCX/CCZ in standard Qiskit)
            if getattr(inst, "definition", None) is not None:
                sub = inst.definition
                qmap = {sub.qubits[i]: qargs[i] for i in range(len(qargs))}
                for s_inst, s_qargs, s_cargs in sub.data:
                    out.append(s_inst, [qmap[q] for q in s_qargs], s_cargs)
                changed = True
                continue

            # ControlledGate with definition path
            if isinstance(inst, ControlledGate) and inst.definition is not None:
                sub = inst.definition
                qmap = {sub.qubits[i]: qargs[i] for i in range(len(qargs))}
                for s_inst, s_qargs, s_cargs in sub.data:
                    out.append(s_inst, [qmap[q] for q in s_qargs], s_cargs)
                changed = True
                continue

            # MCX explicit lowering
            if (name in {"mcx", "mcx_gray", "mcx_recursive", "mcx_vchain"}) or (
                MCXGate is not None and isinstance(inst, MCXGate)
            ):
                num_ctrls = len(qargs) - 1
                decomp = self._lower_mcx(num_ctrls)
                mapping = [qa.index for qa in qargs]  # [controls..., target]
                for s_inst, s_qargs, s_cargs in decomp.data:
                    out.append(s_inst, [out.qubits[mapping[q.index]] for q in s_qargs], s_cargs)
                changed = True
                continue

            # Otherwise, pass through
            out.append(inst, qargs, cargs)

        return self._inline_composites(out) if changed else out

    def _lower_mcx(self, num_ctrls: int) -> QuantumCircuit:
        """Decompose MCX(num_ctrls) using only 1q+CX (noancilla)."""
        if MCXGate is None:
            raise NotImplementedError("MCXGate unavailable in this Qiskit version.")
        gate = MCXGate(num_ctrls=num_ctrls, mode="noancilla")
        sub = QuantumCircuit(num_ctrls + 1)
        sub.append(gate, list(range(num_ctrls + 1)))
        return self._inline_composites(sub)

    # ---------- Basis translation (to {rz,sx,x,cx}) ----------

    def _translate_to_native(self, circ: QuantumCircuit) -> QuantumCircuit:
        """
        Translate a circuit to the native basis:
        - 1q: {rz, sx, x}
        - 2q: {cx}  (cz -> H(t) cx H(t); swap -> 3 cx)
        Assumes 3+ qubit ops were already inlined.
        """
        out = QuantumCircuit(circ.num_qubits)

        for inst, qargs, cargs in circ.data:
            name = inst.name
            if name in {"barrier", "measure", "reset"}:
                # skip non-unitary in translation stage
                continue

            qN = len(qargs)

            if qN == 1:
                q = qargs[0]._index
                if name in self.native_1q:
                    # already native
                    self._emit_1q(out, name, inst.params, q)
                else:
                    # Robust path: let Qiskit lower this single-1q op to {rz, sx, x}
                    from qiskit import transpile
                    tmp = QuantumCircuit(circ.num_qubits)
                    tmp.append(inst, [tmp.qubits[q]], cargs)

                    lowered = transpile(tmp, basis_gates=['rz', 'sx', 'x'], optimization_level=0)

                    for tinst, tqargs, tcargs in lowered.data:
                        if len(tqargs) != 1:
                            raise RuntimeError(
                                f"Unexpected {tinst.name} arity {len(tqargs)} in 1q lowering"
                            )
                        tq = tqargs[0]._index
                        tname = tinst.name
                        if tname in self.native_1q:
                            self._emit_1q(out, tname, getattr(tinst, 'params', []), tq)
                        elif tname in {'u', 'u3'}:
                            # Very old toolchains may still emit 'u' here: lower once more
                            from qiskit.circuit.library import UGate
                            theta, phi, lam = tinst.params
                            tmp2 = QuantumCircuit(circ.num_qubits)
                            tmp2.append(UGate(theta, phi, lam), [tmp2.qubits[tq]])
                            lowered2 = transpile(tmp2, basis_gates=['rz', 'sx', 'x'], optimization_level=0)
                            for ti2, tq2, _ in lowered2.data:
                                self._emit_1q(out, ti2.name, getattr(ti2, 'params', []), tq2[0]._index)
                        else:
                            raise NotImplementedError(
                                f"Lowered 1q emitted unsupported gate '{tname}'"
                            )

            elif qN == 2:
                qc, qt = qargs[0]._index, qargs[1]._index
                if name == "cx":
                    out.cx(qc, qt)
                elif name == "cz":
                    self._emit_H(out, qt); out.cx(qc, qt); self._emit_H(out, qt)
                elif name == "swap":
                    out.cx(qc, qt); out.cx(qt, qc); out.cx(qc, qt)
                else:
                    # Any other 2q should have been inlined already
                    raise NotImplementedError(f"Unsupported 2q gate '{name}' after inlining.")

            else:
                # 3+ qubit ops should have been fully inlined before this stage
                raise NotImplementedError(
                    f"Unsupported {qN}-qubit gate '{name}' after inlining (expected <=2q)."
                )

        return out


    # ---------- 1-qubit synthesis ----------

    def _decompose_1q_to_zsx(self, U) -> List[Tuple[str, List[float]]]:
        """Return a sequence implementing U in ZSX basis as [('rz',[a]), ('sx',[]), ...]."""
        cmds: List[Tuple[str, List[float]]] = []
        try:
            a, b, c = self.decomposer.angles(U)
            cmds = [("rz", [float(a)]), ("sx", []),
                    ("rz", [float(b)]), ("sx", []),
                    ("rz", [float(c)])]
        except Exception:
            sub = self.decomposer(U)  # 1q circuit
            for inst, _, _ in sub.data:
                if inst.name == "rz":
                    cmds.append(("rz", [float(inst.params[0])]))
                elif inst.name == "sx":
                    cmds.append(("sx", []))
                elif inst.name == "x":
                    cmds.append(("x", []))
                elif inst.name in {"id"}:
                    pass
                else:
                    raise NotImplementedError(f"Decomposer emitted unsupported gate '{inst.name}'")
        return cmds

    # ---------- Emitters ----------

    @staticmethod
    def _emit_1q(qc: QuantumCircuit, name: str, params: List, q: int):
        if name == "rz":
            qc.rz(float(params[0]), q)
        elif name == "sx":
            qc.sx(q)
        elif name == "x":
            qc.x(q)
        elif name == "id":
            qc.id(q)
        elif name == "h":
            qc.h(q)
        else:
            raise NotImplementedError(f"Unsupported 1q gate '{name}' in emitter.")

    @staticmethod
    def _emit_H(qc: QuantumCircuit, q: int):
        qc.h(q)

    # ---------- Routing helpers (emit on LOGICAL wires) ----------

    def _append_cx_oriented_LOGICAL(self, out: QuantumCircuit, pc: int, pt: int, layout: Dict[int, int]):
        """Append a CX with the correct orientation, but emit on logical wires per current layout."""
        lc = self._logical_at_physical(layout, pc)
        lt = self._logical_at_physical(layout, pt)
        if (pc, pt) in self.dir_edges:
            out.cx(lc, lt)
        elif (pt, pc) in self.dir_edges:
            self._emit_H(out, lc); self._emit_H(out, lt)
            out.cx(lt, lc)
            self._emit_H(out, lc); self._emit_H(out, lt)
        else:
            out.cx(lc, lt)

    def _emit_swap_via_cx_LOGICAL(self, out: QuantumCircuit, lu: int, lv: int):
        """Emit a SWAP between logical wires lu <-> lv using 3 CX, on the OUTPUT circuit."""
        out.cx(lu, lv)
        out.cx(lv, lu)
        out.cx(lu, lv)


    def _shortest_path(self, s: int, t: int) -> List[int]:
        """Undirected shortest path via BFS (return list of vertices)."""
        if s == t:
            return [s]
        visited = {s}
        parent = {s: None}
        Q = deque([s])
        while Q:
            u = Q.popleft()
            for v in self.adj[u]:
                if v in visited:
                    continue
                visited.add(v)
                parent[v] = u
                if v == t:
                    path = [t]
                    while path[-1] != s:
                        path.append(parent[path[-1]])
                    return list(reversed(path))
                Q.append(v)
        raise RuntimeError(f"No path between qubits {s} and {t} on the coupling graph.")

    def _route_and_emit_cx_LOGICAL(
        self,
        out: QuantumCircuit,
        lc: int,
        lt: int,
        layout: Dict[int, int],
        pi: Dict[int, int],
    ):
        """
        Route a CX between *logical* (lc -> lt).
        We bubble the target's physical location toward the control with conceptual swaps.
        For each hop (u,v):
        - find logicals lu, lv currently at physical u, v (BEFORE swap),
        - emit a LOGICAL swap on wires (pi[lu], pi[lv]),
        - update placement (layout[lu], layout[lv]) and output permutation (pi[lu], pi[lv]).
        Finally emit CX on wires (pi[lc], pi[lt]).
        """
        pc = layout[lc]
        pt = layout[lt]

        if pt in self.adj[pc]:
            out.cx(pi[lc], pi[lt])
            print(f"[route] CX logical {lc} -> {lt}   phys start {pc} -> {pt}   path: [adjacent]")
            print("  [layout] after route:", layout, "  [pi]:", pi)
            return

        path = self._shortest_path(pc, pt)  # e.g., [pc, n1, ..., pt]
        debug_swaps = []

        # bubble target back toward control (reverse along path)
        for i in range(len(path) - 1, 0, -1):
            u, v = path[i - 1], path[i]
            # logicals at these *physical* sites BEFORE swap
            lu = self._logical_at_physical(layout, u)
            lv = self._logical_at_physical(layout, v)
            # emit logical SWAP on the *current wires* for lu, lv
            wu, wv = pi[lu], pi[lv]
            out.cx(wu, wv); out.cx(wv, wu); out.cx(wu, wv)
            debug_swaps.append((u, v, lu, lv, wu, wv))
            # update placement and wire permutation
            layout[lu], layout[lv] = layout[lv], layout[lu]
            pi[lu], pi[lv] = pi[lv], pi[lu]

        # Now adjacent: emit the desired CX on current wires for lc, lt
        out.cx(pi[lc], pi[lt])

        # --- DEBUG ---
        print(f"[route] CX logical {lc} -> {lt}   physical start {pc} -> {pt}   path: {path}")
        for (u, v, lu, lv, wu, wv) in debug_swaps:
            print(f"  [swap] phys ({u},{v}) -> logical swap ({lu},{lv}) on wires ({wu},{wv})")
        print("  [layout] after route:", layout, "  [pi]:", pi)


    def _restore_order_LOGICAL(self, out: QuantumCircuit, layout: Dict[int, int], pi: Dict[int, int], n_log: int):
        """
        Emit logical swaps to bring each logical l to its physical l:
        while layout[l] != l:
            follow a shortest *physical* path and swap neighbors
            at each hop (u,v):
            - lu, lv = logicals currently at physical u, v
            - emit swap on wires (pi[lu], pi[lv])
            - update layout[lu], layout[lv] and pi[lu], pi[lv]
        """
        from collections import deque

        def spath(s: int, t: int):
            if s == t:
                return [s]
            visited = {s}; parent = {s: None}
            Q = deque([s])
            while Q:
                u = Q.popleft()
                for v in self.adj[u]:
                    if v in visited: 
                        continue
                    visited.add(v); parent[v] = u
                    if v == t:
                        path = [t]
                        while path[-1] != s:
                            path.append(parent[path[-1]])
                        return list(reversed(path))
                    Q.append(v)
            raise RuntimeError(f"No path between physical {s} and {t}")

        for l in range(n_log):
            while layout[l] != l:
                p_cur = layout[l]
                p_tgt = l
                path = spath(p_cur, p_tgt)
                for i in range(len(path) - 1):
                    u, v = path[i], path[i + 1]
                    # logicals BEFORE swap
                    lu = self._logical_at_physical(layout, u)
                    lv = self._logical_at_physical(layout, v)
                    # emit swap on current wires
                    wu, wv = pi[lu], pi[lv]
                    out.cx(wu, wv); out.cx(wv, wu); out.cx(wu, wv)
                    # update maps
                    layout[lu], layout[lv] = layout[lv], layout[lu]
                    pi[lu], pi[lv] = pi[lv], pi[lu]
                    if layout[l] == l:
                        break

        # ---------- Debug helper (non-intrusive) ----------
    def debug_stages(self, circ: QuantumCircuit):
        """
        Return intermediate circuits for debugging:
          - unitary: measurements/barriers stripped
          - inlined: after recursive .definition expansion
          - translated: after basis translation to {rz, sx, x, cx}
          - compiled: final output from compile(circ)
        """
        unitary   = self._strip_final_measurements(circ)
        inlined   = self._inline_composites(unitary)
        translated = self._translate_to_native(inlined)
        compiled, layout = self.compile(circ)
        return unitary, inlined, translated, compiled, layout

    # ---------- Utility ----------

    @staticmethod
    def _logical_at_physical(layout: Dict[int, int], p: int) -> int:
        for l, ph in layout.items():
            if ph == p:
                return l
        raise KeyError(f"No logical mapped to physical {p}.")

    @staticmethod
    def _strip_final_measurements(circ: QuantumCircuit) -> QuantumCircuit:
        """Return a copy without measures/barriers/reset for unitary simulation."""
        try:
            return circ.remove_final_measurements(inplace=False)
        except Exception:
            out = QuantumCircuit(circ.num_qubits)
            for inst, qargs, _ in circ.data:
                if inst.name not in {"measure", "barrier", "reset"}:
                    out.append(inst, qargs)
            return out
