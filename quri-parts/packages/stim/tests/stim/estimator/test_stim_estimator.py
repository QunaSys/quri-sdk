# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from concurrent.futures import ThreadPoolExecutor

import pytest

from quri_parts.circuit import H, S
from quri_parts.core.operator import Operator, pauli_label
from quri_parts.core.state import ComputationalBasisState
from quri_parts.stim.estimator import (
    _Estimate,
    create_stim_clifford_concurrent_estimator,
    create_stim_clifford_estimator,
)


def test_estimate_pauli_label() -> None:
    qubit_count = 6

    estimator = create_stim_clifford_estimator()

    p_label = pauli_label("Z2 Z0 Z5")
    state = ComputationalBasisState(qubit_count, bits=0b100101)
    assert estimator(p_label, state).value == -1.0

    p_label = pauli_label("Z2 X0 Z5")
    state_h_appl = state.with_gates_applied([H(0)])
    assert estimator(p_label, state_h_appl).value == -1.0


def test_estimate_operator() -> None:
    qubit_count = 6
    op = Operator(
        {
            pauli_label("Z0"): 1.0,
            pauli_label("Z0 Z2"): 2.0,
            pauli_label("X1 X3 X5"): 3.0,
            pauli_label("Y0 Y2 Y4"): 4.0,
            pauli_label("Z0 Y1 X2"): 5.0,
        }
    )
    estimator = create_stim_clifford_estimator()

    state = ComputationalBasisState(qubit_count, bits=0b0)
    assert estimator(op, state).value == 3.0

    state = ComputationalBasisState(qubit_count, bits=0b101)
    assert estimator(op, state).value == 1.0

    state = ComputationalBasisState(qubit_count, bits=0b101000)
    state_h_appl = state.with_gates_applied([H(1), H(3), H(5)])
    assert estimator(op, state_h_appl).value == 6.0

    state = ComputationalBasisState(qubit_count, bits=0b10101)
    state_g_appl = state.with_gates_applied([H(0), S(0), H(2), S(2), H(4), S(4)])
    assert estimator(op, state_g_appl).value == -4.0

    state = ComputationalBasisState(qubit_count, bits=0b111)
    state_g_appl = state.with_gates_applied([H(1), S(1), H(2)])
    assert estimator(op, state_g_appl).value == -6.0


def test_concurrent_estimate_invalid_arguments() -> None:
    operator = Operator({pauli_label("Z0 Z2 Z5"): 0.1})
    state = ComputationalBasisState(6, bits=0b110010)
    estimator = create_stim_clifford_concurrent_estimator()

    with pytest.raises(ValueError):
        estimator([], [state])

    with pytest.raises(ValueError):
        estimator([operator], [])

    with pytest.raises(ValueError):
        estimator([operator] * 3, [state] * 2)


def test_concurrent_estimate_single_state() -> None:
    ops = [
        Operator({pauli_label("Z0 Z2 Z5"): 1.0}),
        Operator({pauli_label("Z0 Z1"): 1.0j}),
    ]
    state = ComputationalBasisState(6, bits=0b100110)

    with ThreadPoolExecutor(max_workers=2) as executor:
        estimator = create_stim_clifford_concurrent_estimator(executor, concurrency=2)
        result = estimator(ops, [state])

    assert result == [
        _Estimate(value=1.0, error=0.0),
        _Estimate(value=-1.0j, error=0.0),
    ]


def test_concurrent_estimate() -> None:
    ops = [
        Operator({pauli_label("Z0 Z2 Z5"): 1.0}),
        Operator({pauli_label("Z0 Z1"): 1.0j}),
    ]
    states = [
        ComputationalBasisState(6, bits=0b100010),
        ComputationalBasisState(6, bits=0b100111),
    ]

    with ThreadPoolExecutor(max_workers=2) as executor:
        estimator = create_stim_clifford_concurrent_estimator(executor, concurrency=2)
        result = estimator(ops, states)

    assert result == [
        _Estimate(value=-1.0, error=0.0),
        _Estimate(value=1.0j, error=0.0),
    ]
