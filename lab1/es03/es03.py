from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.providers.basic_provider import BasicProvider
from qiskit.qasm2 import dump


qc = QuantumCircuit(2, 2)     # creiamo circuit con 2 qubit e 2 bit 
qc.h([0, 1])          # |++>
qc.cz(1, 0)           # oracle che inverte fase di |11>
#diffuser
qc.h([0, 1]); qc.x([0, 1]); qc.cz(1, 0); qc.x([0, 1]); qc.h([0, 1])  # diffusor
sv = Statevector(qc)  # simulazione state vector
print("Statevector:", sv)  # stampiamo risultato state vector


qc_m = qc.copy() #copia del circuito mentre qc lo lasciamo per essere esportato QASM
qc_m.measure([0, 1], [0, 1]) #misura q0--> c0 e q1-->c1

#eseguiamo circuito sul simulatore
backend = BasicProvider().get_backend("basic_simulator")
res = backend.run(qc_m, shots=1024).result()
print("Counts:", res.get_counts(qc_m)) 


# --- Esporta OpenQASM ---

with open("es03_qasm.qasm", "w") as f:
    dump(qc, f)

print(" File generato con successo!")