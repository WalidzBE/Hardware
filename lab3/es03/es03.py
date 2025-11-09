from qiskit import QuantumCircuit
from qiskit_ibm_runtime.fake_provider import FakeGuadalupeV2
from math import pi
from itertools import product
import numpy as np
from qiskit.quantum_info import Statevector
from collections import deque
from lab3.es02.es02 import translating_to_guadalupe, are_connected, find_shortest_path, reverse_final_swaps, swaps_management

# Studied from https://quantum.cloud.ibm.com/docs/en/guides/transpiler-stages
