from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector
from qiskit.providers.basic_provider import BasicProvider
from qiskit_ibm_runtime.fake_provider import FakeGuadalupeV2
from itertools import product
import json
import numpy as np
import pandas as pd

def import_circuit_from_qasm(file_path):
    return QuantumCircuit.from_qasm_file(file_path)

def print_circuit_info(qc):
    # printing the circuit
    print(qc)

    # printing depth and gate count
    depth = qc.depth()
    gate_counts = qc.count_ops()
    print(f"Depth: {depth}")
    print(f"Gate count: {gate_counts}")

    # # if qc is transpiled return the mapping and logical to physical qubit mapping
    # if qc.layout is not None:
    #     final_layout = qc.layout.final_layout
    #     print("Final layout (logical to physical qubit mapping):", final_layout)
    #     mapping = qc.layout.initial_layout
    #     print("Initial layout (logical to physical qubit mapping):", mapping.get_physical_bits())
    #     # filter "q" Qubits only
        
    return {"depth": depth, "gate_counts": dict(gate_counts)}

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
    # print("Probabilities:", probabilities)
    return probabilities

def print_fidelity_statevectors(sv1,sv2):
    fidelity = np.abs(sv1.inner(sv2))
    print(fidelity)
    return fidelity

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

index = [
    "backend",
    "optimization_level",
    "circuit_depth",
    "gate_counts",
    "fidelities_with_not_compiled_circuit",
]
data = {key: [] for key in index}


qc = import_circuit_from_qasm("lab3/qasm_files/adder_small.qasm")
no_comp_sv = simulate_circuit_statevector(qc)
base_probabilities = simulate_circuit_basic_provider(qc)
base_circuit_info = print_circuit_info(qc)


data["backend"].append("original")
data["optimization_level"].append("original")
data["circuit_depth"].append(base_circuit_info["depth"])
data["gate_counts"].append(json.dumps(base_circuit_info["gate_counts"], default=str))
data["fidelities_with_not_compiled_circuit"].append(1.0)

for optimization_level in range(4):
    qc_t = transpile(qc, basis_gates=['id', 'ry', 'rx', 'rz', 'cx'], optimization_level=optimization_level)
    print(f"Optimization Level {optimization_level}:")
    circuit_info = print_circuit_info(qc_t)
    print()
    comp_sv = simulate_circuit_statevector(qc_t)
    measurement_probabilities = simulate_circuit_basic_provider(qc_t)
    fidelity = print_fidelity_statevectors(no_comp_sv, comp_sv)

    data["backend"].append("TranspilerRyRxRzCX")
    data["optimization_level"].append(optimization_level)
    data["circuit_depth"].append(circuit_info["depth"])
    data["gate_counts"].append(json.dumps(circuit_info["gate_counts"], default=str))
    data["fidelities_with_not_compiled_circuit"].append(float(fidelity))

device_backend = FakeGuadalupeV2()

print("\n--- Backend Properties ---")
get_backend_properties(device_backend)


for optimization_level in range(4):
    qc_t = transpile(qc, backend=device_backend, optimization_level=optimization_level)
    print(f"Optimization Level {optimization_level}:")
    circuit_info = print_circuit_info(qc_t)
    print()
    comp_sv = simulate_circuit_statevector(qc_t)
    measurement_probabilities = simulate_circuit_basic_provider(qc_t)
    # print_fidelity_statevectors(no_comp_sv, comp_sv)
    final_layout = qc_t.layout.final_layout
    print("qubits:", final_layout)
    data["backend"].append(device_backend.name)
    data["optimization_level"].append(optimization_level)
    data["circuit_depth"].append(circuit_info["depth"])
    data["gate_counts"].append(json.dumps(circuit_info["gate_counts"], default=str))
    data["fidelities_with_not_compiled_circuit"].append(float(0)) #todo change this to fidelity value

df = pd.DataFrame(data)
df.to_csv("lab3/es01/adder_circuit_metrics.csv", index=False)
print("Exported circuit metrics to lab3/es01/adder_circuit_metrics.csv")
