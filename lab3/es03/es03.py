from qiskit import QuantumCircuit
from qiskit_ibm_runtime.fake_provider import FakeGuadalupeV2
from math import pi
from itertools import product
import numpy as np
from qiskit.quantum_info import Statevector
from collections import deque


import sys
import os
# Add the path to es02/ folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'es02')))

from es02 import translating_to_guadalupe, are_connected, find_shortest_path, reverse_final_swaps, swaps_management

# Studied from https://quantum.cloud.ibm.com/docs/en/guides/transpiler-stages


#*******************************************
#MAIN
#****************************************

# file_path = "alu-bdd_288.qasm"
file_path = "lab3/qasm_files/alu-bdd_288.qasm"

# caricare un circuito da file QASM
qc = QuantumCircuit.from_qasm_file(file_path)

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
