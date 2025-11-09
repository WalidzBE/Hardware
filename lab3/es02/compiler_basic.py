from qiskit import QuantumCircuit
from qiskit_ibm_runtime.fake_provider import FakeGuadalupeV2
from math import pi
from collections import deque

# -------------------------
# GATE TRANSLATION
# -------------------------
def translating_to_guadalupe(qc_in: QuantumCircuit) -> QuantumCircuit:
    """Translate the input circuit into the native gate set of FakeGuadalupeV2."""
    backend = FakeGuadalupeV2()
    native_gates = backend.configuration().basis_gates
    qc_out = QuantumCircuit(qc_in.num_qubits)
    
    for inst_data in qc_in.data:
        inst = inst_data.operation
        name = inst.name
        qargs = [qc_in.qubits.index(q) for q in inst_data.qubits]

        if name in native_gates:
            qc_out.append(inst, inst_data.qubits, inst_data.clbits)

        elif name == 'h':
            q = qargs[0]
            qc_out.rz(pi/2, q)
            qc_out.sx(q)
            qc_out.rz(pi/2, q)

        elif name == 's':
            qc_out.rz(pi/2, qargs[0])
        elif name in ['sdg', 's_dg', 's†']:
            qc_out.rz(-pi/2, qargs[0])

        elif name == 't':
            qc_out.rz(pi/4, qargs[0])
        elif name in ['tdg', 't_dg', 't†']:
            qc_out.rz(-pi/4, qargs[0])

        elif name == 'cz':
            c, t = qargs
            qc_out.rz(pi, t)
            qc_out.sx(t)
            qc_out.rz(pi, t)
            qc_out.cx(c, t)
            qc_out.rz(pi, t)
            qc_out.sx(t)
            qc_out.rz(pi, t)

        elif name == 'ccx':
            c1, c2, t = qargs
            qc_out.rz(pi/2, t); qc_out.sx(t); qc_out.rz(pi/2, t)
            qc_out.cx(c2, t)
            qc_out.rz(-pi/4, t)
            qc_out.cx(c1, t)
            qc_out.rz(pi/4, t)
            qc_out.cx(c2, t)
            qc_out.rz(-pi/4, t)
            qc_out.cx(c1, t)
            qc_out.rz(pi/4, c2)
            qc_out.rz(pi/4, t)
            qc_out.rz(pi/2, t); qc_out.sx(t); qc_out.rz(pi/2, t)
            qc_out.cx(c1, c2)
            qc_out.rz(pi/4, c1)
            qc_out.rz(-pi/4, c2)
            qc_out.cx(c1, c2)

        elif name == 'u3':
            theta, phi, lam = inst.params
            q = qargs[0]
            qc_out.rz(phi, q)
            qc_out.sx(q)
            qc_out.rz(theta - pi, q)
            qc_out.sx(q)
            qc_out.rz(lam + pi, q)

        else:
            print(f"[WARNING] Gate non gestito: {name}")

    return qc_out


# -------------------------
# CONNECTIVITY + SWAPS
# -------------------------
def are_connected(p1, p2, coupling_map):
    return [p1, p2] in coupling_map or [p2, p1] in coupling_map

def find_shortest_path(coupling_map, start, target):
    graph = {}
    for a, b in coupling_map:
        graph.setdefault(a, set()).add(b)
        graph.setdefault(b, set()).add(a)

    queue = deque([(start, [start])])
    visited = set([start])

    while queue:
        node, path = queue.popleft()
        if node == target:
            return path
        for neigh in graph[node]:
            if neigh not in visited:
                visited.add(neigh)
                queue.append((neigh, path + [neigh]))
    return None

def reverse_final_swaps(qc_mapped):
    mapping = list(range(qc_mapped.num_qubits))
    swaps = []
    for inst_data in qc_mapped.data:
        if inst_data.operation.name == "swap":
            a = qc_mapped.qubits.index(inst_data.qubits[0])
            b = qc_mapped.qubits.index(inst_data.qubits[1])
            mapping[a], mapping[b] = mapping[b], mapping[a]
            swaps.append((a,b))
    qc_reverse = QuantumCircuit(qc_mapped.num_qubits)
    for a,b in reversed(swaps):
        qc_reverse.swap(a,b)
    return qc_mapped.compose(qc_reverse)

def swaps_management(qc_in: QuantumCircuit) -> QuantumCircuit:
    backend = FakeGuadalupeV2()
    coupling_map = backend.configuration().coupling_map  
    num_qubits = qc_in.num_qubits

    logical_to_physical = list(range(num_qubits))
    qc_out = QuantumCircuit(backend.configuration().num_qubits)

    for inst_data in qc_in.data:
        inst = inst_data.operation
        name = inst.name
        qubits = [qc_in.qubits.index(q) for q in inst_data.qubits]

        if name == "cx":
            c_log, t_log = qubits
            c_phys = logical_to_physical[c_log]
            t_phys = logical_to_physical[t_log]

            if not are_connected(c_phys, t_phys, coupling_map):
                path = find_shortest_path(coupling_map, c_phys, t_phys)
                for i in range(len(path) - 2, -1, -1):
                    a, b = path[i], path[i + 1]
                    qc_out.swap(a, b)
                    for log, phys in enumerate(logical_to_physical):
                        if phys == a: logical_to_physical[log] = b
                        elif phys == b: logical_to_physical[log] = a

                c_phys = logical_to_physical[c_log]
                t_phys = logical_to_physical[t_log]

            qc_out.cx(c_phys, t_phys)

        elif len(qubits) == 1:
            q_log = qubits[0]
            q_phys = logical_to_physical[q_log]
            qc_out.append(inst, [q_phys])
        else:
            mapped_qubits = [logical_to_physical[q] for q in qubits]
            qc_out.append(inst, mapped_qubits)

    return qc_out
