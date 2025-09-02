# Licensed under the MIT License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      https://mit-license.org/
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from quri_parts.circuit import NonParametricQuantumCircuit, QuantumCircuit
from quri_parts.circuit.transpile import CircuitTranspiler

from quri_algo.circuit.interface import CircuitFactory
from quri_algo.circuit.utils.transpile import apply_transpiler
from quri_algo.problem import Problem, ProblemT


def fake_transpiler(
    circuit: NonParametricQuantumCircuit,
) -> NonParametricQuantumCircuit:
    return QuantumCircuit(circuit.qubit_count)


class FakeProblem(Problem):
    def __init__(self, n_state_qubit: int):
        self.n_state_qubit = n_state_qubit


class FakeCircuitFactory(CircuitFactory):
    def __init__(
        self, encoded_problem: ProblemT, *, transpiler: CircuitTranspiler | None = None
    ):
        self.qubit_count = 2
        self.encoded_problem = encoded_problem
        self.transpiler = transpiler

    @apply_transpiler  # type: ignore
    def __call__(self) -> NonParametricQuantumCircuit:
        circuit = QuantumCircuit(2)
        circuit.add_H_gate(0)
        circuit.add_CNOT_gate(0, 1)
        return circuit.freeze()


def test_apply_circuit() -> None:
    circuit_factory = FakeCircuitFactory(FakeProblem(1000))
    expected_circuit = QuantumCircuit(2)
    expected_circuit.add_H_gate(0)
    expected_circuit.add_CNOT_gate(0, 1)

    assert circuit_factory() == expected_circuit

    circuit_factory = FakeCircuitFactory(FakeProblem(1000), transpiler=fake_transpiler)
    assert circuit_factory() == QuantumCircuit(2)
