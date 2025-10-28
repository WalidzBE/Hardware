# es02.py
# Validate the basic compiler on one or more OpenQASM files, with stage-by-stage debug
# and a stepwise translator tracer to find the first offending gate.

import argparse
import glob
import os
from collections import Counter

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, state_fidelity
from qiskit_ibm_runtime.fake_provider import FakeGuadalupeV2

from compiler_basic import BasicGuadalupeCompiler


def gate_stats(qc: QuantumCircuit):
    counts = Counter({k: int(v) for k, v in qc.count_ops().items()})
    return qc.depth(), dict(sorted(counts.items(), key=lambda kv: kv[0]))


def load_qasms(paths):
    files = []
    for p in paths:
        if os.path.isdir(p):
            files.extend(glob.glob(os.path.join(p, "*.qasm")))
        else:
            files.append(p)
    return sorted(set(files))


def strip_unitary(qc: QuantumCircuit) -> QuantumCircuit:
    """Return a copy of qc without final measurements/barriers/reset for unitary comparison."""
    try:
        return qc.remove_final_measurements(inplace=False)
    except Exception:
        out = QuantumCircuit(qc.num_qubits)
        for inst, qargs, _ in qc.data:
            if inst.name not in {"measure", "barrier", "reset"}:
                out.append(inst, qargs)
        return out


def translate_single_op(compiler: BasicGuadalupeCompiler, n_qubits: int, inst, qargs, cargs):
    """Translate a single op by wrapping it in a temp circuit and calling _translate_to_native."""
    tmp = QuantumCircuit(n_qubits)
    tmp.append(inst, [tmp.qubits[q._index] for q in qargs], cargs)
    tr = compiler._translate_to_native(tmp)
    return tr


def trace_translate(compiler: BasicGuadalupeCompiler, unitary_ref: QuantumCircuit, inlined: QuantumCircuit,
                    max_print=6) -> None:
    """Walk the inlined circuit gate-by-gate; stop at first translation mismatch."""
    n = inlined.num_qubits
    acc_ref = QuantumCircuit(n)
    acc_tr  = QuantumCircuit(n)

    def last_ops(qc: QuantumCircuit, k: int):
        tail = qc.data[-k:] if len(qc.data) > k else qc.data
        # Deprecation-safe access
        out = []
        for ci in tail:
            inst = getattr(ci, "operation", getattr(ci, "instruction", None)) or ci[0]
            qargs = getattr(ci, "qubits", None)
            if qargs is None:
                qargs = ci[1]
            out.append(f"{inst.name}{tuple(q.index for q in qargs)}")
        return out

    for idx, ci in enumerate(inlined.data, start=1):
        inst = getattr(ci, "operation", getattr(ci, "instruction", None)) or ci[0]
        qargs = getattr(ci, "qubits", None)
        cargs = getattr(ci, "clbits", None)
        if qargs is None:
            qargs, cargs = ci[1], ci[2]

        if inst.name in {"measure", "barrier", "reset"}:
            continue

        acc_ref.append(inst, qargs, cargs)

        tr_op = translate_single_op(compiler, n, inst, qargs, cargs)
        for cj in tr_op.data:
            tinst = getattr(cj, "operation", getattr(cj, "instruction", None)) or cj[0]
            tqargs = getattr(cj, "qubits", None)
            tcargs = getattr(cj, "clbits", None)
            if tqargs is None:
                tqargs, tcargs = cj[1], cj[2]
            acc_tr.append(tinst, tqargs, tcargs)

        F = float(state_fidelity(Statevector.from_instruction(acc_ref),
                                 Statevector.from_instruction(acc_tr)))
        if abs(F - 1.0) > 1e-12:
            print(">> FIRST TRANSLATION MISMATCH at inlined op #{}:".format(idx))
            ps = [float(p) for p in getattr(inst, "params", [])]
            print("   Op: {}  qubits={}  params={}".format(inst.name, [q.index for q in qargs], ps))
            print("   Fidelity so far: {:.10f}".format(F))
            print("   Last {} ops (REF): {}".format(max_print, last_ops(acc_ref, max_print)))
            print("   Last {} ops (TRN): {}".format(max_print, last_ops(acc_tr, max_print)))
            return

    F_final = float(state_fidelity(Statevector.from_instruction(unitary_ref),
                                   Statevector.from_instruction(acc_tr)))
    print(">> Stepwise translation completed with fidelity {:.10f}".format(F_final))


def reorder_to_identity(qc: QuantumCircuit, pi: dict) -> QuantumCircuit:
    """
    Given a circuit 'qc' and a permutation pi: logical l -> output wire pi[l],
    return a new circuit with wires reordered back to identity order for fair comparison.
    We implement the permutation by a minimal SWAP network appended AFTER 'qc'.
    """
    if pi is None:
        return qc

    n = qc.num_qubits
    # Copy the circuit, then append swaps to map wire pi[l] -> l for all l
    out = qc.copy()
    # Build inverse mapping: inv_pi[w] = l such that pi[l] = w
    inv_pi = {w: l for l, w in pi.items()}

    # Bring each l to position l by swapping current wire 'w' with 'l'
    for l in range(n):
        w = inv_pi[l]   # which wire currently holds logical l
        if w == l:
            continue
        # swap wires w <-> l
        out.swap(w, l)
        # update inv_pi because we've moved labels
        lw = inv_pi[l]          # which logical was at l before swap (that's 'l' itself by construction)
        inv_pi[w], inv_pi[l] = inv_pi[l], inv_pi[w]
        # also update pi consistently
        # find logicals whose wires were w and l; swap their pi entries
        l_w = next(ll for ll, ww in pi.items() if ww == w)
        l_l = next(ll for ll, ww in pi.items() if ww == l)
        pi[l_w], pi[l_l] = pi[l_l], pi[l_w]

    return out


def main():
    parser = argparse.ArgumentParser(description="Validate basic compiler on FakeGuadalupeV2.")
    parser.add_argument("paths", nargs="+", help="OpenQASM file(s) or directory(ies)")
    parser.add_argument("--no-restore", action="store_true",
                        help="Do not add final swaps to restore logical order.")
    parser.add_argument("--debug", action="store_true",
                        help="Print intermediate fidelities (after inlining/translation) and stats.")
    parser.add_argument("--trace-translate", action="store_true",
                        help="Gate-by-gate tracing of the translation stage; stops at first mismatch.")
    args = parser.parse_args()

    backend = FakeGuadalupeV2()
    compiler = BasicGuadalupeCompiler(backend, restore_logical_order=not args.no_restore)

    files = load_qasms(args.paths)
    if not files:
        raise SystemExit("No .qasm files found.")

    print("\n=== Exercise 2: Basic compiler validation on FakeGuadalupeV2 ===")
    print(f"Backend qubits: {backend.num_qubits}\n")

    for f in files:
        print(f"--- {f} ---")
        qc = QuantumCircuit.from_qasm_file(f)

        unitary = strip_unitary(qc)
        inlined = compiler._inline_composites(unitary)
        translated = compiler._translate_to_native(inlined)

        # compile now returns (compiled, layout, pi)
        compiled_tuple = compiler.compile(qc)
        if len(compiled_tuple) == 3:
            compiled, final_layout, pi = compiled_tuple
        else:
            compiled, final_layout = compiled_tuple
            pi = None

        if args.debug:
            psi_unit = Statevector.from_instruction(unitary)
            psi_inl  = Statevector.from_instruction(inlined)
            psi_tr   = Statevector.from_instruction(translated)

            # Compare compiled both before and after reordering by pi (if present)
            psi_comp_raw = Statevector.from_instruction(compiled)
            compiled_reordered = reorder_to_identity(compiled, dict(pi) if pi is not None else None)
            psi_comp_ord = Statevector.from_instruction(compiled_reordered)

            F_inlining  = float(state_fidelity(psi_unit, psi_inl))
            F_translate = float(state_fidelity(psi_unit, psi_tr))
            F_compiled_raw = float(state_fidelity(psi_unit, psi_comp_raw))
            F_compiled_ord = float(state_fidelity(psi_unit, psi_comp_ord))

            d_ref, g_ref = gate_stats(unitary)
            d_tr,  g_tr  = gate_stats(translated)
            d_cmp, g_cmp = gate_stats(compiled)

            print(f"Fidelity after inlining:   {F_inlining:.10f}")
            print(f"Fidelity after translate:  {F_translate:.10f}")
            print(f"Fidelity after compile:    {F_compiled_raw:.10f}")
            print(f"Fidelity after reorder(pi):{F_compiled_ord:.10f}")
            print(f"Depths: original={d_ref}  translated={d_tr}  compiled={d_cmp}")
            print(f"Gates (original):   {g_ref}")
            print(f"Gates (translated): {g_tr}")
            print(f"Gates (compiled):   {g_cmp}")
            print(f"Final layout (logical->physical): {final_layout}")
            print(f"Output-wire permutation pi (logical -> wire): {pi}\n")

        if args.trace_translate:
            print("Tracing translation step-by-step to locate first mismatch...")
            trace_translate(compiler, unitary, inlined)


if __name__ == "__main__":
    main()
