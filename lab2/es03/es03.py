from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from pprint import pprint

#calculating number of Iterations
##
# Since we have 2 states and 4 total states, we have:
# N = 4
# M = 2
# Number of iterations = (π/4) * sqrt(N/M) = (π/4) * sqrt(4/2) = (π/4) * sqrt(2)
# which is approximately 1.11, so we round to 2 iterations
# #

# Number of Grover iterations
grover_iters = 2

# Qubit order: [a1, a0, b1, b0]
n = 4

def mc_z(circ, qubits):
    # Apply a controlled-Z over more than 2 qubits using mcx and H gates
    *controls, target = qubits
    circ.h(target)
    circ.mcx(controls, target)
    circ.h(target)

def oracle(circ, qubits):
    # Marking |1011> and |1110>
    for value in ['1011', '1110']:
        for i, bit in enumerate(value):
            if bit == '0':
                circ.x(qubits[i])
        mc_z(circ, qubits)
        for i, bit in enumerate(value):
            if bit == '0':
                circ.x(qubits[i])

def diffuser(circ, qubits):
    # Applying H X mcZ X H
    for q in qubits:
        circ.h(q)
        circ.x(q)
    mc_z(circ, qubits)
    for q in qubits:
        circ.x(q)
        circ.h(q)

qc = QuantumCircuit(n, n)

qubits = [0, 1, 2, 3]  # 0=a1, 1=a0, 2=b1, 3=b0

# Initialize to uniform superposition
for q in qubits:
    qc.h(q)

# Run 2 Grover iterations
for _ in range(grover_iters):
    oracle(qc, qubits)
    diffuser(qc, qubits)

# Measure
qc.measure(range(n), range(n))

print(qc)

backend = AerSimulator()
res = backend.run(qc, shots=1024).result()
counts = res.get_counts(qc)

pprint(counts)
