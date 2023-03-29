# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Sequence

from typing_extensions import TypeAlias

from quri_parts.circuit import (
    LinearMappedUnboundParametricQuantumCircuit,
    Parameter,
    ParameterOrLinearFunction,
)
from quri_parts.circuit.utils.controlled_rotations import add_controlled_RY_gate

#: Alias of ``tuple[int, int]`` which represents the set of orbital indices involved in
#: excitation. The first element is the index of an occupied orbital and the second
#: element is the index of an unoccupied index.
SingleExcitation: TypeAlias = tuple[int, int]

#: Alias of ``tuple[int, int, int, int]`` which represents the set of orbital indices
#: involved in excitation. The first and second element is the indices of occupied
#: orbitals and the third and fourth element is the indices of unoccupied orbitals.
DoubleExcitation: TypeAlias = tuple[int, int, int, int]


def excitations(
    n_spin_orbitals: int, n_fermions: int, delta_sz: float = 0
) -> tuple[Sequence[SingleExcitation], Sequence[DoubleExcitation]]:
    """Returns the list of excitation indices for :attr:`n_spin_orbitals` and
    :attr:`n_fermions` with :attr:`delta_sz` as a tuple of
    :attr:`SingleExcitation` and :class:`DoubleExcitation`."""
    sz = [0.5 if (i % 2 == 0) else -0.5 for i in range(n_spin_orbitals)]
    singles = []
    doubles = []
    for r in range(n_fermions):
        for p in range(n_fermions, n_spin_orbitals):
            if sz[p] - sz[r] == delta_sz:
                singles.append((r, p))
    for s in range(n_fermions - 1):
        for r in range(s + 1, n_fermions):
            for q in range(n_fermions, n_spin_orbitals - 1):
                for p in range(q + 1, n_spin_orbitals):
                    if sz[p] + sz[q] - sz[r] - sz[s] == delta_sz:
                        doubles.append((s, r, q, p))

    return singles, doubles


def add_single_excitation_circuit(
    circuit: LinearMappedUnboundParametricQuantumCircuit,
    excitation_indices: SingleExcitation,
    param_fn: ParameterOrLinearFunction,
) -> LinearMappedUnboundParametricQuantumCircuit:
    r"""Add a particle-conserving single excitation circuit to the given
    :attr:`circuit` implemented as a Givens rotation :math:`G(\theta)`

    .. math::
        \begin{align}
            G(\theta)|01\rangle &= \cos(\theta /2)|01\rangle + \sin(\theta /2)|10
            \rangle \\
            G(\theta)|10\rangle &= \cos(\theta /2)|10\rangle - \sin(\theta /2)|01
            \rangle \\
        \end{align}

    applied to the :attr:`excitation_indices`.
    """
    circuit.add_CNOT_gate(*excitation_indices)
    add_controlled_RY_gate(
        circuit, excitation_indices[1], excitation_indices[0], param_fn
    )
    circuit.add_CNOT_gate(*excitation_indices)
    return circuit


def add_double_excitation_circuit(
    circuit: LinearMappedUnboundParametricQuantumCircuit,
    excitation_indices: DoubleExcitation,
    param_fn: ParameterOrLinearFunction,
) -> LinearMappedUnboundParametricQuantumCircuit:
    r"""Add a particle-conserving double excitation circuit to the given
    :attr:`circuit` implemented as a extended Givens rotation :math:`G^2(\theta)` which
    acts on the space of 4 qubits and performs the :math:`U(2)` rotation on the
    subspace spanned by two states, e.g. :math:`|0011\rangle` and :math:`|1100\rangle`

    .. math::
        \begin{align}
            G^2(\theta)|0011\rangle &= \cos(\theta /2)|0011\rangle + \sin(\theta /2)
            |1100\rangle \\
            G^2(\theta)|1100\rangle &= \cos(\theta /2)|1100\rangle - \sin(\theta /2)
            |0011\rangle \\
        \end{align}

    applied to the :attr:`excitation_indices`.
    """
    if isinstance(param_fn, Parameter):
        p_fn = {param_fn: 0.125}
        inv_sign_p_fn = {param_fn: -0.125}
    else:
        p_fn = {param: 0.125 * val for param, val in param_fn.items()}
        inv_sign_p_fn = {param: -0.125 * val for param, val in param_fn.items()}

    circuit.add_CNOT_gate(excitation_indices[2], excitation_indices[3])
    circuit.add_CNOT_gate(excitation_indices[0], excitation_indices[2])
    circuit.add_H_gate(excitation_indices[3])
    circuit.add_H_gate(excitation_indices[0])
    circuit.add_CNOT_gate(excitation_indices[2], excitation_indices[3])
    circuit.add_CNOT_gate(excitation_indices[0], excitation_indices[1])

    circuit.add_ParametricRY_gate(excitation_indices[1], p_fn)
    circuit.add_ParametricRY_gate(excitation_indices[0], inv_sign_p_fn)

    circuit.add_CNOT_gate(excitation_indices[0], excitation_indices[3])
    circuit.add_H_gate(excitation_indices[3])
    circuit.add_CNOT_gate(excitation_indices[3], excitation_indices[1])

    circuit.add_ParametricRY_gate(excitation_indices[1], p_fn)
    circuit.add_ParametricRY_gate(excitation_indices[0], inv_sign_p_fn)

    circuit.add_CNOT_gate(excitation_indices[2], excitation_indices[1])
    circuit.add_CNOT_gate(excitation_indices[2], excitation_indices[0])

    circuit.add_ParametricRY_gate(excitation_indices[1], inv_sign_p_fn)
    circuit.add_ParametricRY_gate(excitation_indices[0], p_fn)

    circuit.add_CNOT_gate(excitation_indices[3], excitation_indices[1])
    circuit.add_H_gate(excitation_indices[3])
    circuit.add_CNOT_gate(excitation_indices[0], excitation_indices[3])

    circuit.add_ParametricRY_gate(excitation_indices[1], inv_sign_p_fn)
    circuit.add_ParametricRY_gate(excitation_indices[0], p_fn)

    circuit.add_CNOT_gate(excitation_indices[0], excitation_indices[1])
    circuit.add_CNOT_gate(excitation_indices[2], excitation_indices[0])
    circuit.add_H_gate(excitation_indices[0])
    circuit.add_H_gate(excitation_indices[3])
    circuit.add_CNOT_gate(excitation_indices[0], excitation_indices[2])
    circuit.add_CNOT_gate(excitation_indices[2], excitation_indices[3])

    return circuit
