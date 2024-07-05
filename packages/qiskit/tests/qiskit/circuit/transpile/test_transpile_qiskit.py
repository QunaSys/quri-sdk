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
from qiskit.providers.fake_provider import Fake5QV1
from qiskit.transpiler import CouplingMap

from quri_parts.circuit import QuantumCircuit, gate_names, gates
from quri_parts.qiskit.circuit.transpile import QiskitTranspiler


def test_basis_gates() -> None:
    circuit = QuantumCircuit(3)
    circuit.extend(
        [
            gates.H(0),
            gates.X(1),
            gates.S(2),
        ]
    )
    target = QiskitTranspiler(
        basis_gates=[gate_names.X, gate_names.SqrtX, gate_names.RZ, gate_names.CNOT]
    )(circuit)

    expect = QuantumCircuit(3)
    expect.extend(
        [
            gates.RZ(0, np.pi / 2.0),
            gates.SqrtX(0),
            gates.RZ(0, np.pi / 2.0),
            gates.X(1),
            gates.RZ(2, np.pi / 2.0),
        ]
    )
    assert target == expect


def test_optimization() -> None:
    circuit = QuantumCircuit(1)
    circuit.extend(
        [
            gates.H(0),
            gates.H(0),
            gates.T(0),
            gates.X(0),
            gates.X(0),
        ]
    )
    target = QiskitTranspiler(
        basis_gates=[gate_names.H, gate_names.X, gate_names.T], optimization_level=2
    )(circuit)

    expect = QuantumCircuit(1)
    expect.add_T_gate(0)
    assert target == expect


def test_backend() -> None:
    backend = Fake5QV1()

    circuit = QuantumCircuit(3)
    circuit.extend(
        [
            gates.H(0),
            gates.CNOT(0, 1),
            gates.H(1),
            gates.CNOT(1, 2),
            gates.H(2),
            gates.CNOT(2, 0),
        ]
    )
    target = QiskitTranspiler(backend=backend)(circuit)

    coupling_map = CouplingMap(backend.configuration().coupling_map).get_edges()
    for gate in target.gates:
        if gate.name == gate_names.CNOT:
            assert (gate.control_indices[0], gate.target_indices[0]) in coupling_map
