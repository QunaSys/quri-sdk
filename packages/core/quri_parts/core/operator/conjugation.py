# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from quri_parts.circuit import (
    CNOT,
    CZ,
    SWAP,
    H,
    Identity,
    Pauli,
    QuantumGate,
    S,
    Sdag,
    SqrtX,
    SqrtXdag,
    SqrtY,
    SqrtYdag,
    X,
    Y,
    Z,
)
from quri_parts.circuit.gate_names import CLIFFORD_GATE_NAMES

from .pauli import PauliLabel, SinglePauli, pauli_product

# Returns SinglePauli and it's coefficient after the conjugation U*Pauli*U\dag
# by 1 qubit Clifford gates U.
_single_Pauli_1q_Clifford_gate_conjugation_table: dict[
    int, dict[str, tuple[int, float]]
] = {
    SinglePauli.X: {
        X.name: (SinglePauli.X, 1.0),
        Y.name: (SinglePauli.X, -1.0),
        Z.name: (SinglePauli.X, -1.0),
        H.name: (SinglePauli.Z, 1.0),
        S.name: (SinglePauli.Y, 1.0),
        Sdag.name: (SinglePauli.Y, -1.0),
        SqrtX.name: (SinglePauli.X, 1.0),
        SqrtXdag.name: (SinglePauli.X, 1.0),
        SqrtY.name: (SinglePauli.Z, -1.0),
        SqrtYdag.name: (SinglePauli.Z, 1.0),
    },
    SinglePauli.Y: {
        X.name: (SinglePauli.Y, -1.0),
        Y.name: (SinglePauli.Y, 1.0),
        Z.name: (SinglePauli.Y, -1.0),
        H.name: (SinglePauli.Y, -1.0),
        S.name: (SinglePauli.X, -1.0),
        Sdag.name: (SinglePauli.X, 1.0),
        SqrtX.name: (SinglePauli.Z, 1.0),
        SqrtXdag.name: (SinglePauli.Z, -1.0),
        SqrtY.name: (SinglePauli.Y, 1.0),
        SqrtYdag.name: (SinglePauli.Y, 1.0),
    },
    SinglePauli.Z: {
        X.name: (SinglePauli.Z, -1.0),
        Y.name: (SinglePauli.Z, -1.0),
        Z.name: (SinglePauli.Z, 1.0),
        H.name: (SinglePauli.X, 1.0),
        S.name: (SinglePauli.Z, 1.0),
        Sdag.name: (SinglePauli.Z, 1.0),
        SqrtX.name: (SinglePauli.Y, -1.0),
        SqrtXdag.name: (SinglePauli.Y, 1.0),
        SqrtY.name: (SinglePauli.X, 1.0),
        SqrtYdag.name: (SinglePauli.X, -1.0),
    },
}

# Returns SinglePaulis after the conjugation U*Pauli*U\dag by 2 qubit Clifford gates U.
_single_Pauli_2q_Clifford_gate_conjugation_table: dict[
    int, dict[str, dict[str, tuple[int, int]]]
] = {
    SinglePauli.X: {
        CNOT.name: {
            # Pauli acts on control index. return: (control, target)
            "q1": (SinglePauli.X, SinglePauli.X),
            # Pauli acts on target index. return: (control, target)
            "q2": (0, SinglePauli.X),
        },
        CZ.name: {
            # Pauli acts on control index. return: (control, target)
            "q1": (SinglePauli.X, SinglePauli.Z),
            # Pauli acts on target index. return: (control, target)
            "q2": (SinglePauli.Z, SinglePauli.X),
        },
        SWAP.name: {
            # Pauli acts on target1 index. return: (target1, target2)
            "q1": (0, SinglePauli.X),
            # Pauli acts on target2 index. return: (target1, target2)
            "q2": (SinglePauli.X, 0),
        },
    },
    SinglePauli.Y: {
        CNOT.name: {
            "q1": (SinglePauli.Y, SinglePauli.X),
            "q2": (SinglePauli.Z, SinglePauli.Y),
        },
        CZ.name: {
            "q1": (SinglePauli.Y, SinglePauli.Z),
            "q2": (SinglePauli.Z, SinglePauli.Y),
        },
        SWAP.name: {
            "q1": (0, SinglePauli.Y),
            "q2": (SinglePauli.Y, 0),
        },
    },
    SinglePauli.Z: {
        CNOT.name: {
            "q1": (SinglePauli.Z, 0),
            "q2": (SinglePauli.Z, SinglePauli.Z),
        },
        CZ.name: {
            "q1": (SinglePauli.Z, 0),
            "q2": (0, SinglePauli.Z),
        },
        SWAP.name: {
            "q1": (0, SinglePauli.Z),
            "q2": (SinglePauli.Z, 0),
        },
    },
}


def clifford_gate_conjugation(
    gate: QuantumGate, pauli: PauliLabel
) -> tuple[PauliLabel, complex]:
    r"""Returns :class:`PauliLabel` :math:`P'` mapped by :class:`QuantumGate`
    :math:`U` under the conjugation :math:`P'=UPU^{\dagger}`. Note that this
    function only supports Clifford gates.

    In other words, returns the :math:`P'` which satisfies :math:`P'U = UP`
    for given gate :math:`U` and :class:`PauliLabel` :math:`P`.
    """
    if gate.name not in CLIFFORD_GATE_NAMES:
        raise ValueError("{gate.name} gate is not a Clifford gate.")
    elif gate.name == Pauli.name:
        raise NotImplementedError(
            "Please decompose the multi-qubit Pauli gate into single-qubit Pauli gates"
            "before the conjugation. You can use PauliDecomposeTranspiler."
        )
    if gate.name == Identity.name:
        return pauli, 1.0

    res_pauli: PauliLabel = PauliLabel()
    res_coef: complex = 1.0
    # single qubit gates
    if len([*gate.target_indices, *gate.control_indices]) == 1:
        for index, single_pauli in pauli:
            if index == gate.target_indices[0]:
                updated_pauli, coef = _single_Pauli_1q_Clifford_gate_conjugation_table[
                    single_pauli
                ][gate.name]
            else:
                updated_pauli, coef = single_pauli, 1.0
            res_pauli, _ = pauli_product(
                res_pauli, PauliLabel({(index, updated_pauli)})
            )
            res_coef *= coef
    # multi qubit gates (CNOT, CZ, SWAP)
    elif len([*gate.target_indices, *gate.control_indices]) == 2:
        if gate.name == SWAP.name:
            control_index, target_index = gate.target_indices
        else:
            control_index, target_index = (
                *gate.control_indices,
                *gate.target_indices,
            )
        for index, single_pauli in pauli:
            pauli_ctrl = 0
            pauli_target = 0
            updated_pauli_set: set[tuple[int, int]] = set()
            if index not in [control_index, target_index]:
                updated_pauli_set.add((index, single_pauli))

            if index == control_index:
                (
                    pauli_ctrl,
                    pauli_target,
                ) = _single_Pauli_2q_Clifford_gate_conjugation_table[single_pauli][
                    gate.name
                ][
                    "q1"
                ]
            elif index == target_index:
                (
                    pauli_ctrl,
                    pauli_target,
                ) = _single_Pauli_2q_Clifford_gate_conjugation_table[single_pauli][
                    gate.name
                ][
                    "q2"
                ]

            if pauli_ctrl:
                updated_pauli_set.add((control_index, pauli_ctrl))
            if pauli_target:
                updated_pauli_set.add((target_index, pauli_target))
            res_pauli, coef_product = pauli_product(
                res_pauli, PauliLabel(updated_pauli_set)
            )
            res_coef *= coef_product
    return res_pauli, res_coef
