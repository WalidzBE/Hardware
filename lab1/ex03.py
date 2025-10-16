from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.quantum_info import Statevector
from qiskit.providers.basic_provider import BasicProvider
from qiskit.qasm2 import dump

qr = QuantumRegister(2)
cr = ClassicalRegister(2)
qc2 = QuantumCircuit(qr, cr)

qc2.h([0, 1])
qc2.barrier()
qc2.cz(1, 0)
qc2.barrier()
qc2.h([0, 1])
qc2.barrier()
qc2.x([0, 1])
qc2.barrier()
qc2.cz(1, 0)
qc2.barrier()
qc2.x([0, 1])
qc2.barrier()
qc2.h([0, 1])
qc2.barrier()

# if the sizes of quantum and classical registers are the same, we can define measurements with a single line of code
# qc2.measure(qr, cr)
qc2.draw("mpl")

# Circuit simulation with state vector simulator
statevector = Statevector(qc2)
# print the state vector
print(statevector)

# Circuit simulation with basic simulator
backend = BasicProvider().get_backend("basic_simulator")
result = backend.run(qc2).result()

# Print the counts of the results
print(result.get_counts())

with open("es03_qasm.qasm", "w") as f:
    dump(qc2, f)
