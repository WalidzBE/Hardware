
from qiskit import QuantumCircuit
from qiskit_ibm_runtime.fake_provider import FakeGuadalupeV2
from math import pi
from itertools import product
import numpy as np
from qiskit.quantum_info import Statevector
from collections import deque

def translating_to_guadalupe(qc_in: QuantumCircuit) -> QuantumCircuit:
    """Traduce un circuito in gate nativi di FakeGuadalupeV2."""
    
    # Inizializza backend e stampa i gate nativi
    backend = FakeGuadalupeV2()
    native_gates = backend.configuration().basis_gates
    print("Native gates (basis gates):", native_gates)

    # Crea un circuito di output con lo stesso numero di qubit
    qc_out = QuantumCircuit(qc_in.num_qubits)
    
    # Itera su tutte le istruzioni del circuito
    for inst_data in qc_in.data:
        inst = inst_data.operation
        name = inst.name
        qargs = [qc_in.qubits.index(q) for q in inst_data.qubits]  # compatibile con Qiskit ≥1.0

        # --------------------
        # Gate nativi
        # --------------------
        if name in native_gates:
            qc_out.append(inst, inst_data.qubits, inst_data.clbits)
        
        # --------------------
        # Gate non nativi
        # --------------------
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
            # --- decomposizione standard 6 CX ---
           # H(t)
            qc_out.rz(pi/2, t)
            qc_out.sx(t)
            qc_out.rz(pi/2, t)

            # CX(c2, t)
            qc_out.cx(c2, t)

            # Tdg(t) = Rz(-π/4)
            qc_out.rz(-pi/4, t)

            # CX(c1, t)
            qc_out.cx(c1, t)

            # T(t) = Rz(π/4)
            qc_out.rz(pi/4, t)

            # CX(c2, t)
            qc_out.cx(c2, t)

            # Tdg(t)
            qc_out.rz(-pi/4, t)

            # CX(c1, t)
            qc_out.cx(c1, t)

            # T(c2)
            qc_out.rz(pi/4, c2)

            # T(t)
            qc_out.rz(pi/4, t)

            # H(t)
            qc_out.rz(pi/2, t)
            qc_out.sx(t)
            qc_out.rz(pi/2, t)

            # CX(c1, c2)
            qc_out.cx(c1, c2)

            # T(c1)
            qc_out.rz(pi/4, c1)

            # Tdg(c2)
            qc_out.rz(-pi/4, c2)

            # CX(c1, c2)
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


#are neighbours?
def are_connected(p1, p2, coupling_map):
    return [p1, p2] in coupling_map or [p2, p1] in coupling_map
        
from collections import deque
from qiskit import QuantumCircuit
from qiskit_ibm_runtime.fake_provider import FakeGuadalupeV2

def find_shortest_path(coupling_map, start, target):
    """Trova il cammino più corto tra due qubit fisici nella coupling map (BFS)."""
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
    # Applica le SWAP inverse in ordine inverso
    for a,b in reversed(swaps):
        qc_reverse.swap(a,b)
    return qc_mapped.compose(qc_reverse)



def swaps_management(qc_in: QuantumCircuit) -> QuantumCircuit:
    """Gestisce le SWAP in base alla topologia di FakeGuadalupeV2 con routing multiplo."""
    
    backend = FakeGuadalupeV2()
    coupling_map = backend.configuration().coupling_map  
    num_qubits = qc_in.num_qubits

    logical_to_physical = list(range(num_qubits))  # mappatura iniziale 1:1
    num_phys = len(backend.configuration().coupling_map) + 1  # oppure 16
    qc_out = QuantumCircuit(backend.configuration().num_qubits)
    
    print("Mappatura iniziale logico→fisico:", logical_to_physical)

    for inst_data in qc_in.data:
        inst = inst_data.operation
        name = inst.name
        qubits = [qc_in.qubits.index(q) for q in inst_data.qubits]

        # --- CNOT ---
        if name == "cx":
            c_log, t_log = qubits
            c_phys = logical_to_physical[c_log]
            t_phys = logical_to_physical[t_log]

            if not are_connected(c_phys, t_phys, coupling_map):
                # trova percorso più corto tra i due qubit fisici
                path = find_shortest_path(coupling_map, c_phys, t_phys)
                if path is None:
                    raise ValueError(f"Nessun percorso tra {c_phys} e {t_phys}")
                print(f"CX non connesso ({c_phys},{t_phys}) → path {path}")

                # esegui SWAP lungo il percorso (tutti i passaggi tranne l’ultimo)
                for i in range(len(path) - 2, -1, -1):  # swap "portando vicino" il target
                    a, b = path[i], path[i + 1]
                    qc_out.swap(a, b)
                    # aggiorna la mappa logico↔fisico dopo ogni swap
                    for log, phys in enumerate(logical_to_physical):
                        if phys == a:
                            logical_to_physical[log] = b
                        elif phys == b:
                            logical_to_physical[log] = a
                    print(f"  SWAP({a},{b}) → nuova mappa: {logical_to_physical}")

                # aggiorna indici fisici dopo tutte le SWAP
                c_phys = logical_to_physical[c_log]
                t_phys = logical_to_physical[t_log]

            # finalmente esegui la CX corretta
            qc_out.cx(c_phys, t_phys)

        # --- GATE A 1 QUBIT ---
        elif len(qubits) == 1:
            q_log = qubits[0]
            q_phys = logical_to_physical[q_log]
            qc_out.append(inst, [q_phys])

        # --- GATE MULTI-QUBIT (es. CCX già tradotti in CX+RZ+SX) ---
        else:
            mapped_qubits = [logical_to_physical[q] for q in qubits]
            qc_out.append(inst, mapped_qubits)

    print("=== Mapping finale ===")
    for i, p in enumerate(logical_to_physical):
        print(f"Logico q[{i}] → Fisico {p}")

    return qc_out
          
           
#*******************************************
#MAIN
#****************************************


# caricare un circuito da file QASM
qc = QuantumCircuit.from_qasm_file("alu-bdd_288.qasm")

print("Circuito originale:")
print(qc)
print("Depth:", qc.depth())
print("Gate count:", qc.count_ops())


qc_native = translating_to_guadalupe(qc)


print(qc_native)
print(qc_native.count_ops())   


   
#*************************************** 
#TRIVIAL MAPPING
#***************************

backend = FakeGuadalupeV2()  

coupling_map = backend.configuration().coupling_map
print(coupling_map)


#**************************************************
#SWAP MANAGEMENT
#****************************************
qc_mapped = swaps_management(qc_native)

print(qc_mapped)


#*********************************************************
#FIDELITY CONTROL
#***********************************************
def get_probabilities(qc):
    qc_m = qc.copy()
    qc_m.measure_all()
    result = backend.run(qc_m).result()
    counts = result.get_counts()
    num_qubits = qc.num_qubits
    all_states = [''.join(state) for state in product('01', repeat=num_qubits)]
    total = sum(counts.values())
    return np.array([counts.get(s, 0) / total for s in all_states])

# 1. circuito logico originale
qc_original = qc

# 3. circuito mappato (SWAP management)
qc_mapped = swaps_management(qc_native)



# 4. probabilità di misura
p_orig = get_probabilities(qc_original)
p_final = get_probabilities(qc_native)


# 5. calcola la fidelity

#extend original circuit
num_phys = backend.configuration().num_qubits  # 16
qc_original_extended = QuantumCircuit(num_phys)
qc_original_extended.compose(qc_original, inplace=True)


qc_mapped_aligned = reverse_final_swaps(qc_mapped)


F = abs(Statevector.from_instruction(qc_original_extended)
        .inner(Statevector.from_instruction(qc_mapped_aligned)))**2
print("Fidelity (statevector):", F)


















