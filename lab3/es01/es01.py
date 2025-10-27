from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector
from qiskit.providers.basic_provider import BasicProvider
from qiskit_ibm_runtime.fake_provider import FakeGuadalupeV2
from itertools import product
import numpy as np

def import_circuit_from_qasm(file_path):
    return QuantumCircuit.from_qasm_file(file_path)

def print_circuit_info(qc):
        #printing the circuit
    print(qc)

    # printing depth and gate count
    print(f"Depth: {qc.depth()}") 
    print(f"Gate count: {qc.count_ops()}")

def simulate_circuit_statevector(qc):
    sv = Statevector.from_instruction(qc)
    print("Statevector amplitudes:\n", sv)
    print("\nMeasurement probabilities:")
    probs = sv.probabilities_dict()
    for outcome, prob in probs.items():
        print(f"{outcome}: {prob:.4f}")
    return sv

def simulate_circuit_basic_provider(qc):
    # obtaining the vector of probabilities
    qc_m = qc.copy()
    qc_m.measure_all()

    backend = BasicProvider().get_backend("basic_simulator")
    result = backend.run(qc_m).result()
    counts = result.get_counts()
    print("Counts:", counts)
    num_qubits = qc.num_qubits
    # Generate all possible binary outcomes for num_qubits
    all_states = [''.join(state) for state in product('01', repeat=num_qubits)]

    # Calculate probabilities, including zero counts
    total_shots = sum(counts.values())
    probabilities = [counts.get(state, 0) / total_shots for state in all_states]
    print(probabilities)

def print_fidelity_statevectors(sv1,sv2):
    print(np.abs(sv1.inner(sv2)))

def get_backend_properties(backend):
    coupling_map = backend.configuration().coupling_map
    print("Coupling map:", coupling_map)

    properties = backend.properties()
    print("Properties:", properties)

    # Extract calibration data for each qubit
    print("\n--- Qubit Properties ---")
    for qubit_index in range(len(properties.qubits)):
        print(f"Qubit {qubit_index}:")
        for item in properties.qubits[qubit_index]:
            print(f" {item.name}: {item.value} {item.unit}")

    # Extract gate errors for each gate
    print("\n--- Gate Properties ---")
    for gate in properties.gates:
        print(f"Gate: {gate.gate} on qubits {gate.qubits}")
        for param in gate.parameters:
            print(f" {param.name}: {param.value} {param.unit}")
    # Example: Extract T1 and T2 times for each qubit
    print("\n--- Coherence Times ---")
    for qubit_index in range(len(properties.qubits)):
        T1 = properties.t1(qubit_index)
        T2 = properties.t2(qubit_index)
        print(f"Qubit {qubit_index}: T1 = {T1:.2f} s , T2 = {T2:.2f} s ")

    # Example: Extract readout errors for each qubit
    print("\n--- Readout Errors ---")
    for qubit_index in range(len(properties.qubits)):
        readout_error = properties.readout_error(qubit_index)
        print(f"Qubit {qubit_index}: Readout error = {readout_error:.4f}")

    native_gates = backend.configuration().basis_gates
    print("Native gates (basis gates):", native_gates)


#Start of the main code

# backend, optimization_level, circuit depth, gate counts, fidelities_with_not_compiled_circuit
index = ["backend", "optimization_level", "circuit_depth", "gate_counts", "fidelities_with_not_compiled_circuit"]
data = {}


qc = import_circuit_from_qasm("lab3/qasm_files/adder_small.qasm")
no_comp_sv = simulate_circuit_statevector(qc)
simulate_circuit_basic_provider(qc)
print_circuit_info(qc)

for optimization_level in range(4):
    qc_t = transpile(qc, basis_gates=['id', 'ry', 'rx', 'rz', 'cx'], optimization_level=optimization_level)
    print(f"Optimization Level {optimization_level}:")
    print_circuit_info(qc_t)
    print()
    comp_sv = simulate_circuit_statevector(qc_t)
    print_fidelity_statevectors(no_comp_sv, comp_sv)

print("\n--- Backend Properties ---")
backend = FakeGuadalupeV2()
get_backend_properties(backend)

