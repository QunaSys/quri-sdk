# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from quri_parts.chem.utils.excitations import (
    add_double_excitation_circuit,
    add_single_excitation_circuit,
    excitations,
)
from quri_parts.circuit import (
    ImmutableLinearMappedUnboundParametricQuantumCircuit,
    LinearMappedUnboundParametricQuantumCircuit,
)


class AllSinglesDoubles(ImmutableLinearMappedUnboundParametricQuantumCircuit):
    """Parametric quantum circuit consists of single excitation and double
    excitation.

    Note that this circuit conserves the particle number and spins
    only if the state applied to is represented based on the Jordan-Wigner
    transformation.

    Ref:
        PennyLane's documentations,
        `qml.AllSinglesDoubles <https://docs.pennylane.ai/en/stable/code/
        api/pennylane.AllSinglesDoubles.html>`_

    Args:
        n_spin_orbitals: Number of spin orbitals.
        n_fermions: Number of fermions.
    """

    def __init__(
        self,
        n_spin_orbitals: int,
        n_fermions: int,
    ):
        single_excitations, double_excitations = excitations(
            n_spin_orbitals, n_fermions
        )

        circuit = LinearMappedUnboundParametricQuantumCircuit(n_spin_orbitals)
        s_exc_params = circuit.add_parameters(
            *[f"theta_s_{i}" for i in range(len(single_excitations))]
        )
        d_exc_params = circuit.add_parameters(
            *[f"phi_d_{i}" for i in range(len(double_excitations))]
        )

        for d_exc, d_exc_param in zip(double_excitations, d_exc_params):
            add_double_excitation_circuit(circuit, d_exc, d_exc_param)

        for s_exc, s_exc_param in zip(single_excitations, s_exc_params):
            add_single_excitation_circuit(circuit, s_exc, s_exc_param)

        super().__init__(circuit)
