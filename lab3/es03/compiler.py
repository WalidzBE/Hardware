from qiskit import QuantumCircuit
from qiskit_ibm_runtime.fake_provider import FakeGuadalupeV2
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.visualization import circuit_drawer
from qiskit.visualization.dag_visualization import dag_drawer
from math import pi
from itertools import product
import numpy as np
from qiskit.quantum_info import Statevector
from collections import deque
from qiskit.circuit.library import RXGate, RYGate, RZGate, XGate, YGate, ZGate
import math
import matplotlib.pyplot as plt

import sys
import os
# Add the path to es02/ folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'es02')))

from compiler_basic import translating_to_guadalupe, are_connected, find_shortest_path, reverse_final_swaps, swaps_management

def _z_like_angle(name, params):
    if name == "rz":
        return float(params[0])
    if name == "s":
        return math.pi / 2
    if name == "sdg":
        return -math.pi / 2
    if name == "t":
        return math.pi / 4
    if name == "tdg":
        return -math.pi / 4
    return None

def merge_rotations(dag):
    for q in dag.qubits:
        # Ordered list of op nodes acting on this qubit
        nodes = [n for n in dag.topological_op_nodes() if q in n.qargs]
        i = 0
        while i < len(nodes):
            n0 = nodes[i]
            # Identify axis and initial angle
            axis = None
            angle0 = None

            if n0.name == "rx":
                axis, angle0 = "rx", float(n0.op.params[0])
            elif n0.name == "ry":
                axis, angle0 = "ry", float(n0.op.params[0])
            else:
                zang = _z_like_angle(n0.name, getattr(n0.op, "params", []))
                if zang is not None:
                    axis, angle0 = "rz", zang

            # Not a rotation gate we handle → move on
            if axis is None or len(n0.qargs) != 1:
                i += 1
                continue

            # Collect the maximal run of same-axis rotations on the same qubit
            run = [n0]
            total = angle0
            j = i + 1
            while j < len(nodes):
                nj = nodes[j]
                if nj.qargs != n0.qargs or len(nj.qargs) != 1:
                    break

                same_axis = False
                if axis in {"rx", "ry"} and nj.name == axis:
                    same_axis = True
                    angj = float(nj.op.params[0])
                elif axis == "rz":
                    angj = _z_like_angle(nj.name, getattr(nj.op, "params", []))
                    same_axis = (angj is not None)

                if not same_axis:
                    break

                total += angj
                run.append(nj)
                j += 1

            # If the run has only one element, nothing to merge → advance
            if len(run) == 1:
                i += 1
                continue

            # Normalize to (-π, π]
            total = (total + math.pi) % (2 * math.pi) - math.pi

            gate_cls = {"rx": RXGate, "ry": RYGate, "rz": RZGate}[axis]
            new_op = gate_cls(total)
            # Replace the block in-place to preserve ordering around it
            dag.replace_block_with_op(run, new_op, qargs=run[0].qargs, cargs=[])

            # Refresh the nodes list for this qubit and restart near the same spot
            nodes = [n for n in dag.topological_op_nodes() if q in n.qargs]
            # After replacement/removal, the block collapsed to ≤1 node;
            # set i to max(i-1, 0) to catch further adjacent merges
            i = max(i - 1, 0)

    return dag

def remove_double_inverses(dag):
    self_inverse_gates = {"x", "y", "z", "h", "cx", "cz", "swap"}

    # Traverse by topological order
    nodes = list(dag.topological_op_nodes())
    i = 0
    while i < len(nodes) - 1:
        node1 = nodes[i]
        node2 = nodes[i + 1]

        # Check if both are the same self-inverse gate on the same wires
        if (
            node1.name == node2.name
            and node1.name in self_inverse_gates
            and node1.qargs == node2.qargs
            and node1.cargs == node2.cargs
        ):
            dag.remove_op_node(node1)
            dag.remove_op_node(node2)

            # Recompute nodes list since DAG changed
            nodes = list(dag.topological_op_nodes())
            i = 0
            continue

        i += 1

    return dag

def simplify_axis_interactions(dag):
    pauli_gates = {"x", "y", "z"}
    rotation_gates = {"rx", "ry", "rz"}

    gate_classes = {"x": XGate, "y": YGate, "z": ZGate,
                    "rx": RXGate, "ry": RYGate, "rz": RZGate}

    for qubit in dag.qubits:
        nodes = [n for n in dag.topological_op_nodes() if qubit in n.qargs]
        i = 0
        while i < len(nodes) - 1:
            n1 = nodes[i]
            n2 = nodes[i + 1]

            # --- Case 1: Conjugation like X RZ X = RZ(-φ) ---
            if (
                n1.name in pauli_gates
                and n2.name == "rz"
                and i + 2 < len(nodes)
                and nodes[i + 2].name == n1.name
                and all(nodes[i + 2].qargs == n1.qargs)
            ):
                angle = -float(n2.op.params[0])  # flip angle
                new_node = RZGate(angle)
                # remove n1, n2, n3 and insert new RZ
                dag.remove_op_node(nodes[i])
                dag.remove_op_node(nodes[i + 1])
                dag.remove_op_node(nodes[i + 2])
                dag.apply_operation_back(new_node, qargs=n1.qargs)
                nodes = [n for n in dag.topological_op_nodes() if qubit in n.qargs]
                i = 0
                continue

            # --- Case 2: Pauli multiplication XY = Z (up to phase) ---
            if n1.name in pauli_gates and n2.name in pauli_gates:
                if n1.name != n2.name:
                    # mapping table (ignoring global phase)
                    table = {
                        ("x", "y"): "z",
                        ("y", "x"): "z",
                        ("y", "z"): "x",
                        ("z", "y"): "x",
                        ("z", "x"): "y",
                        ("x", "z"): "y",
                    }
                    if (n1.name, n2.name) in table:
                        new_gate = gate_classes[table[(n1.name, n2.name)]]()
                        dag.remove_op_node(n1)
                        dag.remove_op_node(n2)
                        dag.apply_operation_back(new_gate, qargs=n1.qargs)
                        nodes = [n for n in dag.topological_op_nodes() if qubit in n.qargs]
                        i = 0
                        continue

            i += 1

    return dag

def optimize_until_stable(
    dag,
    remove_double_inverses,
    merge_rotations,
    simplify_axis_interactions,
    max_iterations=50,
):

    for iteration in range(max_iterations):
        count_ops_before = dag.count_ops()

        dag_prev = dag

        dag = remove_double_inverses(dag)
        dag = merge_rotations(dag)
        dag = simplify_axis_interactions(dag)

        count_ops_after = dag.count_ops()

        print(f"Iteration {iteration+1}: {count_ops_before} → {count_ops_after} operations")

        if count_ops_after == count_ops_before:
            print("Optimization stabilized.")
            break

    else:
        print("Max iterations reached.")

    return dag
