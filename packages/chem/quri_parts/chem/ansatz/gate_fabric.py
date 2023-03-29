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

import numpy as np

from quri_parts.chem.utils.excitations import add_double_excitation_circuit
from quri_parts.chem.utils.orbital_rotation import add_orbital_rotation_gate
from quri_parts.circuit import (
    CONST,
    ImmutableLinearMappedUnboundParametricQuantumCircuit,
    LinearMappedUnboundParametricQuantumCircuit,
)


class GateFabric(ImmutableLinearMappedUnboundParametricQuantumCircuit):
    """Parametric quantum circuit that conserves the number of particles. Note
    that this circuit conserves the particle number only if the state applied
    to is represented based on the Jordan-Wigner transformation.

    Ref.
        [1] Anselmetti et al.,
        Local, expressive, quantum-number-preserving VQE ansätze for fermionic systems,
        2021 New J. Phys. 23 113010, DOI `10.1088/1367-2630/ac2cb3
        <https://iopscience.iop.org/article/10.1088/1367-2630/ac2cb3>`_

        See also PennyLane's documentations,
        `qml.GateFabric <https://docs.pennylane.ai/en/stable/code/api/
        pennylane.GateFabric.html>`_

    Args:
        n_spin_orbitals: Number of spin orbitals.
        n_layers: Number of layers :math:`D`.
        include_pi: If ``True``, the optional constant \
        :math:`\\hat{\\Pi}=\\textrm{QNP}_{\\textrm{OR}}(\\pi)` \
        gate is inserted. This option does not seem to affect the expressiveness of\
        the quantum circuit, but setting ``True`` can be advantageous during\
        gradient-based parameter optimization [1].
    """

    def __init__(self, n_spin_orbitals: int, n_layers: int, include_pi: bool = False):
        if n_spin_orbitals < 4:
            raise ValueError(
                f"GateFabric ansatz requires at least 4 qubits; got {n_spin_orbitals}."
            )
        circuit = LinearMappedUnboundParametricQuantumCircuit(n_spin_orbitals)

        block_gate_qindices = [
            (i, i + 1, i + 2, i + 3) for i in range(0, n_spin_orbitals - 3, 4)
        ]
        block_gate_qindices += [
            (i, i + 1, i + 2, i + 3) for i in range(2, n_spin_orbitals - 3, 4)
        ]

        for i in range(n_layers):
            for q_indices in block_gate_qindices:
                circuit.extend(_q_gate(n_spin_orbitals, i, q_indices, include_pi))

        super().__init__(circuit)


def _q_gate(
    qubit_count: int,
    layer_index: int,
    qubit_indices: Sequence[int],
    include_pi: bool,
) -> LinearMappedUnboundParametricQuantumCircuit:
    circuit = LinearMappedUnboundParametricQuantumCircuit(qubit_count)
    theta, phi = circuit.add_parameters(
        f"theta_{layer_index}_{qubit_indices[0]}",
        f"phi_{layer_index}_{qubit_indices[0]}",
    )
    if include_pi:
        add_orbital_rotation_gate(circuit, qubit_indices, {CONST: np.pi})
    add_double_excitation_circuit(
        circuit,
        (qubit_indices[0], qubit_indices[1], qubit_indices[2], qubit_indices[3]),
        theta,
    )
    add_orbital_rotation_gate(circuit, qubit_indices, phi)

    return circuit
