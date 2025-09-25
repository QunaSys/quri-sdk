# Licensed under the MIT License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      https://mit-license.org/
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from typing import cast

import numpy as np
import numpy.typing as npt
from quri_parts.circuit import NonParametricQuantumCircuit, QuantumCircuit
from quri_parts.circuit.transpile import CircuitTranspiler
from quri_parts.core.operator import get_sparse_matrix
from scipy.linalg import expm

from quri_algo.circuit.time_evolution.interface import (
    ControlledTimeEvolutionCircuitFactory,
    TimeEvolutionCircuitFactory,
)
from quri_algo.circuit.utils.transpile import apply_transpiler
from quri_algo.problem.operators.hamiltonian import QubitHamiltonian


class ExactUnitaryTimeEvolutionCircuitFactory(TimeEvolutionCircuitFactory):
    """Time evolution with exact unitary matrix."""

    def __init__(
        self,
        qubit_hamiltonian: QubitHamiltonian,
        *,
        transpiler: CircuitTranspiler | None = None
    ):
        self.qubit_hamiltonian = qubit_hamiltonian
        self._hamiltonian_matrix = get_sparse_matrix(
            self.qubit_hamiltonian.qubit_hamiltonian
        ).toarray()
        self.transpiler = transpiler

    def get_time_evo_matrix(self, evolution_time: float) -> npt.NDArray[np.complex128]:
        time_evo = expm(-1j * evolution_time * self._hamiltonian_matrix)

        return cast(npt.NDArray[np.complex128], time_evo)

    @apply_transpiler  # type: ignore
    def __call__(self, evolution_time: float) -> NonParametricQuantumCircuit:
        unitary = self.get_time_evo_matrix(evolution_time).tolist()
        n_state_qubits = self.qubit_hamiltonian.n_qubit
        circuit = QuantumCircuit(n_state_qubits)
        circuit.add_UnitaryMatrix_gate(range(n_state_qubits), unitary)
        return circuit


class ExactUnitaryControlledTimeEvolutionCircuitFactory(
    ControlledTimeEvolutionCircuitFactory
):
    """Controlled time evolution with exact unitary matrix."""

    def __init__(
        self,
        qubit_hamiltonian: QubitHamiltonian,
        *,
        transpiler: CircuitTranspiler | None = None
    ):
        self.qubit_hamiltonian = qubit_hamiltonian
        self.qubit_count = qubit_hamiltonian.n_qubit + 1
        self.transpiler = transpiler
        self._hamiltonian_matrix = get_sparse_matrix(
            self.qubit_hamiltonian.qubit_hamiltonian
        ).toarray()

    def get_controlled_time_evo_matrix(
        self, evolution_time: float
    ) -> npt.NDArray[np.complex128]:
        time_evo = expm(-1j * evolution_time * self._hamiltonian_matrix)
        unitary = np.kron(
            np.eye(2**self.qubit_hamiltonian.n_qubit, dtype=np.complex128),
            np.array([[1, 0], [0, 0]], dtype=np.complex128),
        )
        unitary += np.kron(time_evo, np.array([[0, 0], [0, 1]], dtype=np.complex128))

        return cast(npt.NDArray[np.complex128], unitary)

    @apply_transpiler  # type: ignore
    def __call__(self, evolution_time: float) -> NonParametricQuantumCircuit:
        unitary = self.get_controlled_time_evo_matrix(evolution_time).tolist()
        n_state_qubits = self.qubit_hamiltonian.n_qubit
        circuit = QuantumCircuit(n_state_qubits + 1)
        circuit.add_UnitaryMatrix_gate(range(n_state_qubits + 1), unitary)
        return circuit
