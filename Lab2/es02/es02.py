# es02.py  —  2-bit adder (no Cin), Basic Simulator, QASM export

from qiskit import QuantumCircuit
from qiskit.providers.basic_provider import BasicProvider
from qiskit.qasm2 import dump

#  c[2]=c2, c[1]=s1, c[0]=s0   -->  Qiskit prints "c2 s1 s0"

def prepare_inputs(qc: QuantumCircuit, A: int, B: int):
    """Encode A=a1a0 and B=b1b0 (0..3) into |a1 a0 b1 b0>."""
    if A & 0b10: qc.x(0)  # a1
    if A & 0b01: qc.x(1)  # a0
    if B & 0b10: qc.x(2)  # b1
    if B & 0b01: qc.x(3)  # b0

def add_2bit(qc: QuantumCircuit):
    """Reversible 2-bit adder without carry-in."""
    # --- LSB half adder ---
    # c1 = a0 AND b0  -> qubit 4
    qc.ccx(1, 3, 4)
    # s0 = a0 XOR b0  -> qubit 7 (clean target)
    qc.cx(1, 7)
    qc.cx(3, 7)

    # --- MSB full adder with c1 ---
    # s1 = a1 XOR b1 -> qubit 6
    qc.cx(0, 6)
    qc.cx(2, 6)

    # c2 = (a1 AND b1) OR (c1 AND (a1 XOR b1)) -> qubit 5
    qc.ccx(0, 2, 5)   # add a1 b1 term
    qc.ccx(6, 4, 5)   # add c1*(a1 XOR b1) term

    # finalize s1 = (a1 XOR b1) XOR c1
    qc.cx(4, 6)

def measure_sum(qc: QuantumCircuit):
    """Measure into classical bits so printed order is 'c2 s1 s0'."""
    qc.measure(7, 0)  # s0 -> c[0]
    qc.measure(6, 1)  # s1 -> c[1]
    qc.measure(5, 2)  # c2 -> c[2]

def run_and_counts(qc: QuantumCircuit, shots=1024):
    backend = BasicProvider().get_backend("basic_simulator")
    result = backend.run(qc, shots=shots).result()
    return result.get_counts(qc)

def verify_all():
    print("Truth table A(2b)+B(2b) -> c2 s1 s0  (decimal check)")
    ok = True
    for A in range(4):
        for B in range(4):
            qc = QuantumCircuit(8, 3)
            prepare_inputs(qc, A, B)
            add_2bit(qc)
            measure_sum(qc)
            counts = run_and_counts(qc, shots=1)  # deterministic
            bitstring = next(iter(counts))  # order is 'c2 s1 s0'
            c2 = int(bitstring[0]); s1 = int(bitstring[1]); s0 = int(bitstring[2])
            got = (c2 << 2) | (s1 << 1) | s0
            exp = A + B
            flag = "OK" if got == exp else "!!"
            print(f"A={A:02b} B={B:02b}  ->  {c2}{s1}{s0}  == {got}  (exp {exp}) {flag}")
            ok &= (got == exp)
    print("Verification:", "PASS" if ok else "FAIL")
    return ok

def submission_circuit():
    # Required submission inputs: A=01, B=11
    A, B = 0b01, 0b11
    qc = QuantumCircuit(8, 3)
    prepare_inputs(qc, A, B)
    add_2bit(qc)

    # Save unmeasured circuit for OpenQASM export
    qc_unmeasured = qc.copy()

    # Measure and simulate
    measure_sum(qc)
    counts = run_and_counts(qc, shots=1024)
    print("Submission counts (A=01, B=11):", counts)

    # Export OpenQASM 2.0
    with open("adder2q_A01_B11.qasm", "w") as f:
        dump(qc_unmeasured, f)
    print("Wrote QASM to adder2q_A01_B11.qasm")
    return qc, counts

if __name__ == "__main__":
    verify_all()
    submission_circuit()
