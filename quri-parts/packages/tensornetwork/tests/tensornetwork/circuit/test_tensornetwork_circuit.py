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
import tensornetwork as tn
from numpy.testing import assert_almost_equal

from quri_parts.circuit import QuantumCircuit
from quri_parts.circuit.gates import CNOT, H, X, Y, Z
from quri_parts.tensornetwork.circuit import convert_circuit

circuit_tensor_pairs = [
    (
        QuantumCircuit(1, gates=[X(0)]),
        np.array([[0.0, 1.0], [1.0, 0.0]]),
    ),
    (
        QuantumCircuit(1, gates=[Y(0)]),
        np.array([[0.0, 1.0j], [-1.0j, 0.0]]),
    ),
    (
        QuantumCircuit(1, gates=[Z(0)]),
        np.array([[1.0, 0.0], [0.0, -1.0]]),
    ),
    (
        QuantumCircuit(2, gates=[H(0), CNOT(0, 1)]),
        np.array(
            [
                [
                    [
                        [1 / np.sqrt(2), 0.0],
                        [0.0, 1 / np.sqrt(2)],
                    ],
                    [
                        [0.0, 1 / np.sqrt(2)],
                        [1 / np.sqrt(2), 0.0],
                    ],
                ],
                [
                    [
                        [1 / np.sqrt(2), 0.0],
                        [0.0, -1 / np.sqrt(2)],
                    ],
                    [
                        [0.0, 1 / np.sqrt(2)],
                        [-1 / np.sqrt(2), 0.0],
                    ],
                ],
            ]
        ),
    ),
]


def test_convert_circuit() -> None:
    for c, t in circuit_tensor_pairs:
        tensornetwork_circuit = convert_circuit(c)
        all_edges = list(tensornetwork_circuit.input_edges) + list(
            tensornetwork_circuit.output_edges
        )
        contracted_state = tn.contractors.optimal(
            tensornetwork_circuit._container, output_edge_order=all_edges
        )
        assert_almost_equal(contracted_state.tensor, t)
