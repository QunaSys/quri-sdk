from typing import Any, Sequence

from quri_parts.core.operator import PAULI_IDENTITY, trotter_suzuki_decomposition
from quri_parts.qsub.lib.std import PauliRotation
from quri_parts.qsub.op import Op
from quri_parts.qsub.opsub import ParamUnitarySubDef, param_opsub
from quri_parts.qsub.qubit import Qubit
from quri_parts.qsub.sub import SubBuilder

from quri_algo.problem import QubitHamiltonian


class _TrotterTimeEvo(ParamUnitarySubDef[QubitHamiltonian, float, int, int]):
    name = "TrotterTimeEvolution"

    def qubit_count_fn(self, h: QubitHamiltonian, *_: Any) -> int:
        return h.n_qubit

    def sub(
        self,
        builder: SubBuilder,
        h: QubitHamiltonian,
        t: float,
        n_trotter: int = 1,
        order: int = 1,
    ) -> None:
        if order == 1:
            hamiltonian_items = h.qubit_hamiltonian.items()
            pauli_coef_pairs = [
                (op, 2 * t * coeff / n_trotter) for op, coeff in hamiltonian_items
            ]
        else:
            assert order % 2 == 0, "Trotter order must be 1 or an even number."
            exp_list = trotter_suzuki_decomposition(
                h.qubit_hamiltonian, 2 * t / n_trotter, order // 2
            )
            pauli_coef_pairs = [(op, coeff) for op, coeff in exp_list]
        op_q_list: list[tuple[Op, Sequence[Qubit]]] = []
        for p, c in pauli_coef_pairs:
            if p == PAULI_IDENTITY:
                continue
            else:
                qubits, pauli_ids = p.index_and_pauli_id_list
                op_q_list.append(
                    (
                        PauliRotation(tuple(pauli_ids), c),
                        tuple(builder.qubits[q] for q in qubits),
                    )
                )
        for _ in range(n_trotter):
            for op, q in op_q_list:
                builder.add_op(op, q)


TrotterTimeEvo, TrotterTimeEvoSub = param_opsub(_TrotterTimeEvo)
