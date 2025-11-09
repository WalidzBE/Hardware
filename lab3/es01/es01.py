import json
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from qiskit import QuantumCircuit, transpile
from qiskit.providers.basic_provider import BasicProvider
from qiskit.quantum_info import Statevector
from qiskit.visualization import plot_circuit_layout
from qiskit_ibm_runtime.fake_provider import FakeGuadalupeV2

def import_circuit_from_qasm(file_path):
    return QuantumCircuit.from_qasm_file(file_path)

def print_circuit_info(qc):
    print()
    print("CIRCUIT INFO:")
    # printing the circuit
    print(qc)

    # printing depth and gate count
    depth = qc.depth()
    gate_counts = qc.count_ops()
    print(f"Depth: {depth}")
    print(f"Gate count: {gate_counts}")

    return {"depth": depth, "gate_counts": dict(gate_counts)}

def simulate_circuit_statevector(qc):
    print()
    print("STATEVECTOR SIMULATION:")
    sv = Statevector.from_instruction(qc)
    print("Statevector amplitudes:\n", sv)
    print("\nMeasurement probabilities:")
    probs = sv.probabilities_dict()
    for outcome, prob in probs.items():
        print(f"{outcome}: {prob:.4f}")
    return sv

def simulate_circuit_basic_provider(qc):
    print()
    print("BASIC PROVIDER SIMULATION:")
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

def reorder_qubits(qc_t):
    # print()
    # print("REORDER QUBITS:")
    # Print the transpiled circuit and its final layout
    # print("Original transpiled circuit:")
    # print(qc_t)

    final_virtual_layout = qc_t.layout.final_virtual_layout(filter_ancillas=True)
    # print("\nFinal virtual layout:")
    # print(final_virtual_layout)

    # Create a mapping from logical qubits (virtual) to physical qubits
    layout_map = {final_virtual_layout[phys]._index: phys for phys in final_virtual_layout.get_physical_bits()}
    # print("\nLayout map (virtual -> physical):", layout_map)

    num_qubits = len(final_virtual_layout)
    new_qc = QuantumCircuit(num_qubits)

    # Reorder: map physical qubits back to logical order
    qubit_map = {phys: virt for virt, phys in layout_map.items()}

    # Copy gates from the transpiled circuit into the new circuit with reordered qubits
    for inst, qargs, cargs in qc_t.data:
        new_qargs = [new_qc.qubits[qubit_map[q._index]] for q in qargs]
        new_qc.append(inst, new_qargs, cargs)

    # print("\nReordered circuit:")
    # print(new_qc)
    return new_qc

def get_fidelity_statevectors(sv1,sv2):
    print()
    print("Fidelity:")
    fidelity = np.abs(sv1.inner(sv2))
    print(fidelity)
    return fidelity

def print_fidelity_statevectors_from_qc(qc1, qc2):

    sv1 = Statevector.from_instruction(qc1)
    sv2 = Statevector.from_instruction(qc2)

    fidelity = get_fidelity_statevectors(sv1,sv2)

    return fidelity

def get_backend_properties(backend):
    print()
    print("BACKEND PROPERTIES")

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

def collect_metrics_for_qasm(qasm_path):
    print("\n" + "-" * 80)
    print(f"Processing QASM file: {qasm_path.name}")
    print("-" * 80)

    data_rows = []

    qc = import_circuit_from_qasm(str(qasm_path))
    no_comp_sv = simulate_circuit_statevector(qc)
    base_probabilities = simulate_circuit_basic_provider(qc)
    base_circuit_info = print_circuit_info(qc)

    data_rows.append(
        {
            "qasm_file": qasm_path.name,
            "backend": "original",
            "optimization_level": "original",
            "circuit_depth": base_circuit_info["depth"],
            "gate_counts": json.dumps(base_circuit_info["gate_counts"], default=str),
            "fidelities_with_not_compiled_circuit": 1.0,
        }
    )

    for optimization_level in range(4):
        qc_t = transpile(
            qc, basis_gates=["id", "ry", "rx", "rz", "cx"], optimization_level=optimization_level
        )
        circuit_info = print_circuit_info(qc_t)
        measurement_probabilities = simulate_circuit_basic_provider(qc_t)
        fidelity = print_fidelity_statevectors_from_qc(qc, qc_t)
        data_rows.append(
            {
                "qasm_file": qasm_path.name,
                "backend": "TranspilerRyRxRzCX",
                "optimization_level": optimization_level,
                "circuit_depth": circuit_info["depth"],
                "gate_counts": json.dumps(circuit_info["gate_counts"], default=str),
                "fidelities_with_not_compiled_circuit": float(fidelity),
            }
        )

    device_backend = FakeGuadalupeV2()
    # get_backend_properties(device_backend)

    for optimization_level in range(4):
        qc_t = transpile(qc, backend=device_backend, optimization_level=optimization_level)
        print(f"Optimization Level {optimization_level}:")
        circuit_info = print_circuit_info(qc_t)
        measurement_probabilities = simulate_circuit_basic_provider(qc_t)
        reordered_qc_t = reorder_qubits(qc_t)
        fidelity = print_fidelity_statevectors_from_qc(qc, reordered_qc_t)
        print("FIDELITY:", fidelity)
        data_rows.append(
            {
                "qasm_file": qasm_path.name,
                "backend": device_backend.name,
                "optimization_level": optimization_level,
                "circuit_depth": circuit_info["depth"],
                "gate_counts": json.dumps(circuit_info["gate_counts"], default=str),
                "fidelities_with_not_compiled_circuit": float(fidelity),
            }
        )
        fig = plot_circuit_layout(qc_t, device_backend)
        plt.show()

    return data_rows

def write_metrics(grouped_results):
    rows = []
    for qasm_name, data_rows in grouped_results:
        if rows:
            rows.append({column: "" for column in RESULT_COLUMNS})
        label_row = {column: "" for column in RESULT_COLUMNS}
        label_row["qasm_file"] = f"QASM file: {qasm_name}"
        rows.append(label_row)

        rows.extend(row.copy() for row in data_rows)

    df = pd.DataFrame(rows, columns=RESULT_COLUMNS)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Exported circuit metrics to {OUTPUT_PATH}")

if __name__ == "__main__":
    #Start of the main code

    RESULT_COLUMNS = [
        "qasm_file",
        "backend",
        "optimization_level",
        "circuit_depth",
        "gate_counts",
        "fidelities_with_not_compiled_circuit",
    ]

    CURRENT_DIR = Path(__file__).resolve().parent
    QASM_DIR = CURRENT_DIR.parent / "qasm_files"
    OUTPUT_PATH = CURRENT_DIR / "es01.csv"


    qasm_files = sorted(QASM_DIR.glob("*.qasm"))
    if not qasm_files:
        raise FileNotFoundError(f"No .qasm files found in {QASM_DIR}")

    results = []
    for qasm_file in qasm_files:
        results.append((qasm_file.name, collect_metrics_for_qasm(qasm_file)))

    write_metrics(results)
