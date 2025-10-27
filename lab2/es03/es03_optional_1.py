from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from pprint import pprint

#calculating number of Iterations
##
# Since we have 1 state and 8 total states, we have:
# N = 8
# M = 1
# Number of iterations = (π/4) * sqrt(N/M) = (π/4) * sqrt(8/1) = (π/4) * sqrt(8)
# which is approximately 2.22, so we round to 3 iterations
# #

#Calculating the marking state
## depending on the conditions on a,b,c we have:
# a+c = 2 then (0,b,2), (2,b,0)
# b+c = 3 then (0,1,2), (2,3,0)
# a+b = 5 leaving only (2,3,0)
# thus the only state that satisfies all conditions is |101100> (a=2,b=3,c=0)
##


# Number of Grover iterations
grover_iters = 3

# Qubit order: [a1, a0, b1, b0, c1, c0]
n = 6

def mc_z(circ, qubits):
    # Apply a controlled-Z over more than 2 qubits using mcx and H gates
    *controls, target = qubits
    circ.h(target)
    circ.mcx(controls, target)
    circ.h(target)

def oracle(circ, qubits):
    # Marking |101100>
    for value in ['101100']:
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

qubits = [0, 1, 2, 3, 4, 5]  # 0=a1, 1=a0, 2=b1, 3=b0, 4=c1, 5=c0


# Initialize to uniform superposition
for q in qubits:
    qc.h(q)

# Run 3 Grover iterations
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