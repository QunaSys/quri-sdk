# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np

from quri_parts.circuit import (
    ImmutableLinearMappedUnboundParametricQuantumCircuit,
    LinearMappedUnboundParametricQuantumCircuit,
    Parameter,
    ParameterOrLinearFunction,
)
from quri_parts.circuit.utils.controlled_rotations import add_controlled_RY_gate


class ParticleConservingU1(ImmutableLinearMappedUnboundParametricQuantumCircuit):
    """Parametric quantum circuit that conserves the number of particles.

    Note that this circuit conserves the particle number and spins only
    if the state applied to is represented based on the Jordan-Wigner
    transformation.

    Ref.
        P. Kl. Barkoutsos et al.,
        Quantum algorithms for electronic structure calculations: particle/hole
        Hamiltonian and optimized wavefunction expansions,
        Phys. Rev. A 98, 022322

        See also PennyLane's documentations,
        `qml.ParticleConservingU1 <https://docs.pennylane.ai/en/stable/code/api/
        pennylane.ParticleConservingU1.html>`_

    Args:
        n_spin_orbitals: Number of spin orbitals.
        n_layers: Number of layers :math:`D`.
    """

    def __init__(self, n_spin_orbitals: int, n_layers: int):
        circuit = LinearMappedUnboundParametricQuantumCircuit(n_spin_orbitals)

        block_gate_qindices = [(i, i + 1) for i in range(0, n_spin_orbitals - 1, 2)]
        block_gate_qindices += [(i, i + 1) for i in range(1, n_spin_orbitals - 1, 2)]

        for i in range(n_layers):
            for qidx1, qidx2 in block_gate_qindices:
                circuit.extend(_u1_ex_gate(n_spin_orbitals, i, qidx1, qidx2))

        super().__init__(circuit)


def _u1_ex_gate(
    qubit_count: int, layer_index: int, qubit_index_1: int, qubit_index_2: int
) -> LinearMappedUnboundParametricQuantumCircuit:
    circuit = LinearMappedUnboundParametricQuantumCircuit(qubit_count)
    phi, theta = circuit.add_parameters(
        f"phi_{layer_index}_{qubit_index_1}", f"theta_{layer_index}_{qubit_index_1}"
    )
    _add_controlled_ua_gate(circuit, qubit_index_1, qubit_index_2, phi)
    circuit.add_CZ_gate(qubit_index_2, qubit_index_1)
    add_controlled_RY_gate(circuit, qubit_index_2, qubit_index_1, {theta: 2.0})
    _add_controlled_ua_gate(circuit, qubit_index_1, qubit_index_2, {phi: -1.0})

    return circuit


def _add_controlled_ua_gate(
    circuit: LinearMappedUnboundParametricQuantumCircuit,
    control_index: int,
    target_index: int,
    param_fn: ParameterOrLinearFunction,
) -> LinearMappedUnboundParametricQuantumCircuit:
    if isinstance(param_fn, Parameter):
        inv_sign_p_fn = {param_fn: -1.0}
    else:
        inv_sign_p_fn = {param: -1.0 * val for param, val in param_fn.items()}

    circuit.add_CZ_gate(control_index, target_index)

    circuit.add_ParametricRZ_gate(target_index, inv_sign_p_fn)
    circuit.add_CNOT_gate(control_index, target_index)
    circuit.add_RY_gate(target_index, -np.pi / 2)
    circuit.add_CNOT_gate(control_index, target_index)
    circuit.add_RY_gate(target_index, np.pi / 2)
    circuit.add_ParametricRZ_gate(target_index, param_fn)

    circuit.add_ParametricRZ_gate(target_index, inv_sign_p_fn)
    circuit.add_CNOT_gate(control_index, target_index)
    circuit.add_ParametricRZ_gate(target_index, param_fn)
    circuit.add_CNOT_gate(control_index, target_index)
    circuit.add_ParametricRZ_gate(control_index, inv_sign_p_fn)

    return circuit
