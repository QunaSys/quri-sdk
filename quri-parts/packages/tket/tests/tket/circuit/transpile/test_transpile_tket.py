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

from quri_parts.circuit import QuantumCircuit, gate_names, gates
from quri_parts.tket.circuit.transpile import TketTranspiler


def test_basis_gates() -> None:
    circuit = QuantumCircuit(3)
    circuit.extend(
        [
            gates.H(0),
            gates.X(1),
            gates.S(2),
        ]
    )
    target = TketTranspiler(
        basis_gates=[gate_names.X, gate_names.SqrtX, gate_names.RZ, gate_names.CNOT],
        optimization_level=1,
    )(circuit)

    expect = QuantumCircuit(3)
    expect.extend(
        [
            gates.RZ(0, np.pi / 2.0),
            gates.X(1),
            gates.RZ(2, np.pi / 2.0),
            gates.SqrtX(0),
            gates.RZ(0, np.pi / 2.0),
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
    target = TketTranspiler(
        basis_gates=[gate_names.X, gate_names.SqrtX, gate_names.RZ, gate_names.CNOT],
        optimization_level=2,
    )(circuit)

    expect = QuantumCircuit(1)
    expect.add_RZ_gate(0, np.pi / 4.0)
    assert target == expect
