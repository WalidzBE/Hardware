
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.providers.basic_provider import BasicProvider
from qiskit.qasm2 import dump

def apply_oracle_single_mark(qc: QuantumCircuit, mark: str):

    if mark not in {"00", "01", "10", "11"}:
        raise ValueError("mark must be one of {'00','01','10','11'}")
    if mark[1] == '0': 
        qc.x(0)
    if mark[0] == '0': 
        qc.x(1)

    qc.cz(1, 0)  

    if mark[0] == '0':
        qc.x(1)
    if mark[1] == '0':
        qc.x(0)

def apply_diffuser_2q(qc: QuantumCircuit):

    qc.h([0, 1])
    qc.x([0, 1])
    qc.cz(1, 0)
    qc.x([0, 1])
    qc.h([0, 1])

def build_grover_2q(mark: str, r: int) -> QuantumCircuit:
 
    if r not in (1, 2):
        raise ValueError("r must be 1 or 2.")
    qc = QuantumCircuit(2, 2)
    qc.h([0, 1])
    for _ in range(r):
        apply_oracle_single_mark(qc, mark)
        apply_diffuser_2q(qc)
    return qc


def main():
    # Ask oracle
    mark = input("Choose oracle (one to mark) among 00, 01, 10, 11: ").strip()
    if mark not in {"00", "01", "10", "11"}:
        print("Invalid choice. Defaulting to '11'.")
        mark = "11"

    try:
        r = int(input("Number of Grover iterations (1 or 2): ").strip())
    except Exception:
        r = 1
    if r not in (1, 2):
        print("Invalid iteration count. Defaulting to 1.")
        r = 1

    # Build circuit
    qc = build_grover_2q(mark, r)


    sv = Statevector(qc)
    print("Statevector (unmeasured circuit):")
    print(sv)

    qc_m = qc.copy()
    qc_m.measure([0, 1], [0, 1])

    backend = BasicProvider().get_backend("basic_simulator")
    res = backend.run(qc_m, shots=1024).result()
    print("Counts (1024 shots):", res.get_counts(qc_m))

    qasm_filename = f"grover2q_{mark}_r{r}.qasm"
    with open(qasm_filename, "w") as f:
        dump(qc, f)
    print(f"QASM written to {qasm_filename}")

if __name__ == "__main__":
    main()

