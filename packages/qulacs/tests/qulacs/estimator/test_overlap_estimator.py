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

import pytest
from numpy.testing import assert_almost_equal

from quri_parts.circuit import QuantumGate
from quri_parts.circuit.gates import CNOT, H
from quri_parts.core.state import ComputationalBasisState
from quri_parts.qulacs.overlap_estimator import (
    create_qulacs_vector_overlap_estimator,
    create_qulacs_vector_overlap_weighted_sum_estimator,
)


class TestOverlapEstimator:
    def test_estimate_same_state(self) -> None:
        state = ComputationalBasisState(6, bits=0b110010)
        estimator = create_qulacs_vector_overlap_estimator()
        estimate = estimator(state, state)
        assert estimate.value == 1.0
        assert estimate.error == 0.0

    def test_estimate_half(self) -> None:
        ket = ComputationalBasisState(6, bits=0b110010)
        h = H(0)
        bra = ket.with_gates_applied([h])
        estimator = create_qulacs_vector_overlap_estimator()
        estimate = estimator(ket, bra)
        assert_almost_equal(estimate.value, 0.5)
        assert estimate.error == 0.0


class TestOverlapWeightedSumEstimator:
    def test_invalid_arguments(self) -> None:
        state = ComputationalBasisState(6, bits=0b110010)
        estimator = create_qulacs_vector_overlap_weighted_sum_estimator(None, 1)

        with pytest.raises(ValueError):
            estimator([state], [], [])

        with pytest.raises(ValueError):
            estimator([state], [state], [])

        with pytest.raises(ValueError):
            estimator([state], [], [0.1])

        with pytest.raises(ValueError):
            estimator([state], [state], [0.1, 0.1])

    def test_overlap_weighted_sum(self) -> None:
        basis_state = ComputationalBasisState(2)
        ket_gate_sets: Sequence[Sequence[QuantumGate]] = [
            [H(0)],
            [],
            [H(1)],
        ]
        bra_gate_sets: Sequence[Sequence[QuantumGate]] = [
            [H(0), H(1)],
            [H(0), CNOT(0, 1)],
            [H(1)],
        ]
        weights = [2.0, 3.0, 5.0]
        kets = [basis_state.with_gates_applied(gates) for gates in ket_gate_sets]
        bras = [basis_state.with_gates_applied(gates) for gates in bra_gate_sets]
        estimator = create_qulacs_vector_overlap_weighted_sum_estimator(None, 1)
        estimate = estimator(kets, bras, weights)

        assert_almost_equal(estimate.value, 7.5)
        assert estimate.error == 0.0
