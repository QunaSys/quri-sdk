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

from quri_parts.circuit import (
    ImmutableLinearMappedUnboundParametricQuantumCircuit,
    LinearMappedUnboundParametricQuantumCircuit,
)
from quri_parts.circuit.utils.controlled_rotations import add_controlled_RX_gate


class ParticleConservingU2(ImmutableLinearMappedUnboundParametricQuantumCircuit):
    """Parametric quantum circuit that conserves the number of particles. Note
    that this circuit conserves the particle number and spins only if the state
    applied to is represented based on the Jordan-Wigner transformation.

    Ref.
        P. Kl. Barkoutsos et al.,
        Quantum algorithms for electronic structure calculations: particle/hole
        Hamiltonian and optimized wavefunction expansions,
        Phys. Rev. A 98, 022322

        See also PennyLane's documentations,
        `qml.ParticleConservingU2 <https://docs.pennylane.ai/en/latest/code/api/
        pennylane.ParticleConservingU2.html>`_

    Args:
        n_spin_orbitals: Number of spin orbitals.
        n_layers: Number of layers :math:`D`.
    """

    def __init__(self, n_spin_orbitals: int, n_layers: int):
        circuit = LinearMappedUnboundParametricQuantumCircuit(n_spin_orbitals)

        block_gate_qindices = [(i, i + 1) for i in range(0, n_spin_orbitals - 1, 2)]
        block_gate_qindices += [(i, i + 1) for i in range(1, n_spin_orbitals - 1, 2)]

        for i in range(n_layers):
            for j in range(n_spin_orbitals):
                theta = circuit.add_parameter(f"theta_{i}_{j}")
                circuit.add_ParametricRZ_gate(j, {theta: 1.0})
            for qidx1, qidx2 in block_gate_qindices:
                circuit.extend(_u2_ex_gate(n_spin_orbitals, i, (qidx1, qidx2)))

        super().__init__(circuit)


def _u2_ex_gate(
    qubit_count: int, layer_index: int, qubit_indices: Sequence[int]
) -> LinearMappedUnboundParametricQuantumCircuit:
    circuit = LinearMappedUnboundParametricQuantumCircuit(qubit_count)
    phi = circuit.add_parameter(f"phi_{layer_index}_{qubit_indices[0]}")
    circuit.add_CNOT_gate(qubit_indices[0], qubit_indices[1])

    add_controlled_RX_gate(circuit, qubit_indices[1], qubit_indices[0], {phi: 2.0})

    circuit.add_CNOT_gate(qubit_indices[0], qubit_indices[1])

    return circuit
