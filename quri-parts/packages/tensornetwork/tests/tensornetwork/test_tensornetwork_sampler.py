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
from numpy.random import default_rng
from numpy.testing import assert_almost_equal

from quri_parts.circuit import QuantumCircuit
from quri_parts.circuit.gates import CNOT, H, S
from quri_parts.tensornetwork.sampler import (
    create_tensornetwork_ideal_sampler,
    create_tensornetwork_sampler,
)

NSHOTS = 10000
SEED = 42
circuit_probabilities_pairs = [
    (QuantumCircuit(1, gates=[]), [1.0, 0.0]),
    (QuantumCircuit(2, gates=[H(0), CNOT(0, 1)]), [1.0 / 2.0, 0.0, 0.0, 1.0 / 2.0]),
    (
        QuantumCircuit(2, gates=[H(0), H(1), S(1)]),
        [1.0 / 4.0, 1.0 / 4.0, 1.0 / 4.0, 1.0 / 4.0],
    ),
    (QuantumCircuit(2, gates=[H(0)]), [1.0 / 2.0, 1.0 / 2.0, 0.0, 0.0]),
    (
        QuantumCircuit(3, gates=[H(0)]),
        [1.0 / 2.0, 1.0 / 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ),
    (
        QuantumCircuit(3, gates=[H(2)]),
        [1.0 / 2.0, 0.0, 0.0, 0.0, 1.0 / 2.0, 0.0, 0.0, 0.0],
    ),
]


def test_ideal_sampler() -> None:
    sampler = create_tensornetwork_ideal_sampler()
    for c, p in circuit_probabilities_pairs:
        counts = sampler(c, NSHOTS)
        count_list = [counts[b] / NSHOTS for b in range(2**c.qubit_count)]
        assert_almost_equal(count_list, p)


def test_sampler() -> None:
    sampler = create_tensornetwork_sampler(SEED)
    for c, p in circuit_probabilities_pairs:
        rng = default_rng(SEED)
        counts = sampler(c, NSHOTS)
        count_list = [counts[b] / NSHOTS for b in range(2**c.qubit_count)]
        rp = np.round(p, 12)
        norm = np.sum(rp)
        assert_almost_equal(count_list, rng.multinomial(NSHOTS, rp / norm) / NSHOTS)
