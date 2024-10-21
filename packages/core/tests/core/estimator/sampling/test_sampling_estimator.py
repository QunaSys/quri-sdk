# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest
from collections.abc import Collection, Iterable
from math import sqrt
from typing import Any, Union, cast
from unittest.mock import Mock

import numpy as np
import pytest

from quri_parts.circuit import (
    H,
    ImmutableQuantumCircuit,
    ParametricQuantumCircuit,
    QuantumCircuit,
    X,
)
from quri_parts.core.estimator import Estimate
from quri_parts.core.estimator.sampling import (
    concurrent_sampling_estimate,
    create_general_sampling_estimator,
    create_sampling_concurrent_estimator,
    create_sampling_estimator,
    get_estimate_from_sampling_result,
    get_sampling_circuits_and_shots,
    sampling_estimate,
)
from quri_parts.core.measurement import (
    CachedMeasurementFactory,
    CommutablePauliSetMeasurement,
    CommutablePauliSetMeasurementTuple,
    bitwise_commuting_pauli_measurement,
    bitwise_pauli_reconstructor_factory,
)
from quri_parts.core.operator import (
    PAULI_IDENTITY,
    CommutablePauliSet,
    Operator,
    PauliLabel,
    pauli_label,
    zero,
)
from quri_parts.core.sampling import MeasurementCounts, PauliSamplingSetting
from quri_parts.core.sampling.shots_allocator import (
    create_equipartition_shots_allocator,
)
from quri_parts.core.state import (
    CircuitQuantumState,
    ComputationalBasisState,
    ParametricCircuitQuantumState,
)

n_qubits = 3


def initial_state() -> ComputationalBasisState:
    return ComputationalBasisState(n_qubits, bits=0b101)


def state_prep_circuit() -> QuantumCircuit:
    circuit = QuantumCircuit(n_qubits)
    circuit.add_X_gate(0)
    circuit.add_X_gate(2)
    return circuit


def counts() -> MeasurementCounts:
    return {
        0b000: 1,
        0b001: 1,
        0b100: 2,
        0b111: 4,
    }


def total_shots() -> int:
    return cast(int, sum(counts().values()) * 4)


def sampler(
    shot_circuit_pairs: Iterable[tuple[ImmutableQuantumCircuit, int]]
) -> Iterable[MeasurementCounts]:
    return [counts() for _ in shot_circuit_pairs]


def mock_sampler() -> Mock:
    mock = Mock()
    mock.side_effect = sampler
    return mock


def measurement_factory(
    op: Union[Operator, Iterable[PauliLabel]]
) -> Iterable[CommutablePauliSetMeasurement]:
    return [
        CommutablePauliSetMeasurementTuple(
            pauli_set=frozenset(pauli_set),
            measurement_circuit=circuit,
            pauli_reconstructor_factory=bitwise_pauli_reconstructor_factory,
        )
        for pauli_set, circuit in [
            ({PAULI_IDENTITY}, tuple()),
            ({pauli_label("Z0"), pauli_label("Z1")}, tuple()),
            ({pauli_label("X0 Z1")}, (H(0),)),
            ({pauli_label("X1 Z2")}, (H(1),)),
            ({pauli_label("Z0 X1 X2")}, (H(1), H(2))),
        ]
    ]


def sampled_circuits() -> list[QuantumCircuit]:
    return [
        state_prep_circuit(),
        state_prep_circuit() + [H(0)],
        state_prep_circuit() + [H(1)],
        state_prep_circuit() + [H(1), H(2)],
    ]


def operator() -> Operator:
    return Operator(
        {
            PAULI_IDENTITY: 1.0 + 1.0j,
            pauli_label("Z0"): 2.0,
            pauli_label("Z1"): -1.0j,
            pauli_label("X0 Z1"): -1.0 + 2.0j,
            pauli_label("X1 Z2"): 3.0 + -2.0j,
            pauli_label("Z0 X1 X2"): 4.0,
        }
    )


allocator = create_equipartition_shots_allocator()


def assert_sampler_args(s: Mock) -> None:
    s.assert_called_once()
    args: list[Any] = list(*s.call_args.args)
    assert len(args) == 4
    expected_circuits = sampled_circuits()
    for circuit, shots in args:
        assert shots == total_shots() // 4
        assert circuit in expected_circuits


def assert_sample(estimate: Estimate[complex]) -> None:
    c = 1.0 + 1.0j
    v000 = 2.0 - 1.0j + (-1.0 + 2.0j) + (3.0 + -2.0j) + 4.0 + c  # 0b000
    v001 = -2.0 - 1.0j - (-1.0 + 2.0j) + (3.0 + -2.0j) - 4.0 + c  # 0b001
    v100 = 2.0 - 1.0j + (-1.0 + 2.0j) - (3.0 + -2.0j) - 4.0 + c  # 0b100
    v111 = -2.0 + 1.0j + (-1.0 + 2.0j) + (3.0 + -2.0j) - 4.0 + c  # 0b111

    expected_exp = (v000 + v001 + 2 * v100 + 4 * v111) / 8
    assert estimate.value == expected_exp

    expected_var = (
        abs(v000 - expected_exp) ** 2
        + abs(v001 - expected_exp) ** 2
        + 2 * abs(v100 - expected_exp) ** 2
        + 4 * abs(v111 - expected_exp) ** 2
    ) / 8
    expected_err = sqrt(expected_var / 8)
    assert estimate.error == expected_err


def test_get_estimate_from_sampling_result() -> None:
    op = operator()
    measurement_groups = measurement_factory(op)
    sampling_counts = [counts() for _ in measurement_groups]

    estimate = get_estimate_from_sampling_result(
        op, measurement_groups, 0, sampling_counts
    )
    assert_sample(estimate)


class TestSamplingEstimate:
    def test_zero_op(self) -> None:
        estimate = sampling_estimate(
            zero(), initial_state(), 1000, sampler, measurement_factory, allocator
        )
        assert estimate.value == 0.0
        assert estimate.error == 0.0

    def test_raise_when_not_estimatable(self) -> None:
        with pytest.raises(
            AssertionError,
            match="Number of qubits of the operator is too large to estimate.",
        ):
            sampling_estimate(
                pauli_label("Z3"),
                initial_state(),
                1000,
                sampler,
                measurement_factory,
                allocator,
            )
        with pytest.raises(
            AssertionError,
            match="Number of qubits of the operator is too large to estimate.",
        ):
            sampling_estimate(
                Operator({pauli_label("Z3"): 3, pauli_label("X0 X1"): 3}),
                initial_state(),
                1000,
                sampler,
                measurement_factory,
                allocator,
            )

    def test_const_op(self) -> None:
        op = Operator({PAULI_IDENTITY: 3.0})
        estimate = sampling_estimate(
            op, initial_state(), 1000, sampler, measurement_factory, allocator
        )
        assert estimate.value == 3.0
        assert estimate.error == 0.0

    def test_sampling_estimate(self) -> None:
        op = operator()
        s = mock_sampler()
        estimate = sampling_estimate(
            op, initial_state(), total_shots(), s, measurement_factory, allocator
        )
        assert_sampler_args(s)
        assert_sample(estimate)

    def test_cached_sampling_estimate(self) -> None:
        op = operator()
        s = mock_sampler()
        mock_measurement_factory = Mock(side_effect=bitwise_commuting_pauli_measurement)
        cached_measurement_factory = CachedMeasurementFactory(mock_measurement_factory)

        # repeat estimation twice to test if the function uses cahced result
        for _ in range(2):
            estimate = sampling_estimate(
                op,
                initial_state(),
                total_shots(),
                s,
                cached_measurement_factory,
                allocator,
            )
            assert_sample(estimate)
            mock_measurement_factory.assert_called_once()
            assert len(cached_measurement_factory.cached_groups) == 1

    def test_sample_with_customized_circuit_shot_prep_function(self) -> None:
        def prep(
            state: CircuitQuantumState,
            measurement_groups: Iterable[CommutablePauliSetMeasurement],
            shots_map: dict[CommutablePauliSet, int],
        ) -> Iterable[tuple[ImmutableQuantumCircuit, int]]:
            default_pairs = get_sampling_circuits_and_shots(
                state, measurement_groups, shots_map
            )
            new_pairs = []
            for circuit, shot in default_pairs:
                new_circuit = circuit.combine([X(0)])
                new_pairs.append((new_circuit, shot))
            return new_pairs

        op = operator()
        s = mock_sampler()
        sampling_estimate(
            op,
            initial_state(),
            total_shots(),
            s,
            bitwise_commuting_pauli_measurement,
            allocator,
            prep,
        )
        args: list[Any] = list(*s.call_args.args)
        expected_circuits = [c.combine([X(0)]) for c in sampled_circuits()]

        for circuit, shots in args:
            assert shots == total_shots() // 4
            assert circuit in expected_circuits

    def test_sampling_estimate_zero_shots(self) -> None:
        def sampler(
            shot_circuit_pairs: Iterable[tuple[ImmutableQuantumCircuit, int]]
        ) -> Iterable[MeasurementCounts]:
            return [{} if s == 0 else counts() for _, s in shot_circuit_pairs]

        def allocator(
            op: Operator, pauli_sets: Collection[CommutablePauliSet], total_shots: int
        ) -> Collection[PauliSamplingSetting]:
            s = total_shots // (len(pauli_sets) - 1)
            return [
                PauliSamplingSetting(pauli_set, 0 if i == 0 else s)
                for i, pauli_set in enumerate(pauli_sets)
            ]

        op = operator()
        estimate = sampling_estimate(
            op, initial_state(), total_shots(), sampler, measurement_factory, allocator
        )
        assert isinstance(estimate.value, complex)


class TestSamplingEstimator:
    def test_sampling_estimator(self) -> None:
        s = mock_sampler()
        estimator = create_sampling_estimator(
            total_shots(), s, measurement_factory, allocator
        )
        estimate = estimator(operator(), initial_state())
        assert_sampler_args(s)
        assert_sample(estimate)

    def test_raises_when_not_estimatable(self) -> None:
        s = mock_sampler()
        estimator = create_sampling_estimator(
            total_shots(), s, measurement_factory, allocator
        )
        with pytest.raises(
            AssertionError,
            match="Number of qubits of the operator is too large to estimate.",
        ):
            estimator(pauli_label("Z3"), initial_state())

        with pytest.raises(
            AssertionError,
            match="Number of qubits of the operator is too large to estimate.",
        ):
            estimator(
                Operator({pauli_label("Z3"): 3, pauli_label("X0 X1"): 3}),
                initial_state(),
            )


class TestConcurrentSamplingEstimate:
    def test_invalid_arguments(self) -> None:
        s = mock_sampler()

        with pytest.raises(ValueError):
            concurrent_sampling_estimate(
                [], [initial_state()], total_shots(), s, measurement_factory, allocator
            )

        with pytest.raises(ValueError):
            concurrent_sampling_estimate(
                [operator()], [], total_shots(), s, measurement_factory, allocator
            )

        with pytest.raises(ValueError):
            concurrent_sampling_estimate(
                [operator()] * 3,
                [initial_state()] * 2,
                total_shots(),
                s,
                measurement_factory,
                allocator,
            )

        with pytest.raises(
            AssertionError,
            match="Number of qubits of the operator is too large to estimate.",
        ):
            obs1 = pauli_label("Z3")
            obs2 = Operator({pauli_label("Z3"): 3, pauli_label("X0 X1"): 3})
            concurrent_sampling_estimate(
                [obs1, obs2],
                [initial_state()] * 2,
                total_shots(),
                s,
                measurement_factory,
                allocator,
            )

    def test_concurrent_estimate(self) -> None:
        s = mock_sampler()

        estimates = concurrent_sampling_estimate(
            [operator(), pauli_label("Z0")],
            [initial_state(), ComputationalBasisState(3, bits=0b001)],
            total_shots(),
            s,
            bitwise_commuting_pauli_measurement,
            allocator,
        )

        estimate_list = list(estimates)
        assert len(estimate_list) == 2
        assert_sample(estimate_list[0])
        assert estimate_list[1].value == (1 - 1 + 2 - 4) / 8

    def test_cached_concurrent_estimate(self) -> None:
        s = mock_sampler()

        op1 = operator()
        op2 = pauli_label("Z0")
        cached_measurement_factory = CachedMeasurementFactory(
            bitwise_commuting_pauli_measurement
        )

        estimates = concurrent_sampling_estimate(
            [op1, op2],
            [initial_state(), ComputationalBasisState(3, bits=0b001)],
            total_shots(),
            s,
            cached_measurement_factory,
            allocator,
        )

        estimate_list = list(estimates)
        assert len(estimate_list) == 2
        assert_sample(estimate_list[0])
        assert estimate_list[1].value == (1 - 1 + 2 - 4) / 8

    def test_concurrent_estimate_single_state(self) -> None:
        s = mock_sampler()

        estimates = concurrent_sampling_estimate(
            [operator(), pauli_label("Z0")],
            [initial_state()],
            total_shots(),
            s,
            bitwise_commuting_pauli_measurement,
            allocator,
        )

        estimate_list = list(estimates)
        assert len(estimate_list) == 2
        assert_sample(estimate_list[0])
        assert estimate_list[1].value == (1 - 1 + 2 - 4) / 8

    def test_cached_concurrent_estimate_single_state(self) -> None:
        s = mock_sampler()
        op1 = operator()
        op2 = pauli_label("Z0")
        mock_measurement_factory = Mock(side_effect=bitwise_commuting_pauli_measurement)

        cached_measurement_factory = CachedMeasurementFactory(mock_measurement_factory)

        # repeat estimation twice to test if the function uses cahced result
        for _ in range(2):
            estimates = concurrent_sampling_estimate(
                [op1, op2],
                [initial_state()],
                total_shots(),
                s,
                cached_measurement_factory,
                allocator,
            )

            estimate_list = list(estimates)
            assert len(estimate_list) == 2
            assert_sample(estimate_list[0])
            assert estimate_list[1].value == (1 - 1 + 2 - 4) / 8
            assert len(cached_measurement_factory.cached_groups) == 2
            assert mock_measurement_factory.call_count == 2

    def test_concurrent_estimate_single_operator(self) -> None:
        s = mock_sampler()

        estimates = concurrent_sampling_estimate(
            [operator()],
            [initial_state(), ComputationalBasisState(3, bits=0b001)],
            total_shots(),
            s,
            bitwise_commuting_pauli_measurement,
            allocator,
        )

        estimate_list = list(estimates)
        assert len(estimate_list) == 2
        assert_sample(estimate_list[0])
        assert_sample(estimate_list[1])

    def test_cached_concurrent_estimate_single_operator(self) -> None:
        s = mock_sampler()
        op = operator()
        mock_measurement_factory = Mock(side_effect=bitwise_commuting_pauli_measurement)
        cached_measurement_factory = CachedMeasurementFactory(mock_measurement_factory)

        # repeat estimation twice to test if the function uses cahced result
        for _ in range(2):
            estimates = concurrent_sampling_estimate(
                [op],
                [initial_state(), ComputationalBasisState(3, bits=0b001)],
                total_shots(),
                s,
                cached_measurement_factory,
                allocator,
            )

            estimate_list = list(estimates)
            assert len(estimate_list) == 2
            assert_sample(estimate_list[0])
            assert_sample(estimate_list[1])
            assert len(cached_measurement_factory.cached_groups) == 1
            mock_measurement_factory.assert_called_once()


class TestSamplingConcurrentEstimator:
    def test_sampling_concurrent_estimator(self) -> None:
        s = mock_sampler()

        estimator = create_sampling_concurrent_estimator(
            total_shots(),
            s,
            bitwise_commuting_pauli_measurement,
            allocator,
        )
        estimates = estimator(
            [operator(), pauli_label("Z0")],
            [initial_state(), ComputationalBasisState(3, bits=0b001)],
        )

        estimate_list = list(estimates)
        assert len(estimate_list) == 2
        assert_sample(estimate_list[0])
        assert estimate_list[1].value == (1 - 1 + 2 - 4) / 8


class GeneralSamplingEstimator(unittest.TestCase):
    def setUp(self) -> None:
        s = mock_sampler()
        self.general_estimator = create_general_sampling_estimator(
            total_shots(),
            s,
            bitwise_commuting_pauli_measurement,
            allocator,
        )

    def test_general_quantum_estimator(self) -> None:
        estimate = self.general_estimator(operator(), initial_state())
        assert_sample(estimate)

    def test_concurrent_estimate(self) -> None:
        estimates = self.general_estimator(
            operator(),
            [initial_state(), ComputationalBasisState(3, bits=0b001)],
        )

        estimate_list = list(estimates)
        assert len(estimate_list) == 2
        assert_sample(estimate_list[0])
        assert_sample(estimate_list[1])

        estimates = self.general_estimator(
            [operator()],
            [initial_state(), ComputationalBasisState(3, bits=0b001)],
        )

        estimate_list = list(estimates)
        assert len(estimate_list) == 2
        assert_sample(estimate_list[0])
        assert_sample(estimate_list[1])

        estimates = self.general_estimator(
            [operator(), pauli_label("Z0")],
            ComputationalBasisState(3, bits=0b001),
        )

        estimate_list = list(estimates)
        assert len(estimate_list) == 2
        assert_sample(estimate_list[0])
        assert estimate_list[1].value == (1 - 1 + 2 - 4) / 8

        estimates = self.general_estimator(
            [operator(), pauli_label("Z0")],
            [ComputationalBasisState(3, bits=0b001)],
        )

        estimate_list = list(estimates)
        assert len(estimate_list) == 2
        assert_sample(estimate_list[0])
        assert estimate_list[1].value == (1 - 1 + 2 - 4) / 8

        estimates = self.general_estimator(
            [operator(), pauli_label("Z0")],
            [initial_state(), ComputationalBasisState(3, bits=0b001)],
        )

        estimate_list = list(estimates)
        assert len(estimate_list) == 2
        assert_sample(estimate_list[0])
        assert estimate_list[1].value == (1 - 1 + 2 - 4) / 8

    def test_parametric_estimate(self) -> None:
        circuit = ParametricQuantumCircuit(n_qubits)
        circuit.add_X_gate(0)
        circuit.add_ParametricRX_gate(0)
        circuit.add_ParametricRY_gate(1)
        circuit.add_ParametricRZ_gate(2)

        state = ParametricCircuitQuantumState(n_qubits, circuit)

        estimate = self.general_estimator(operator(), state, [0, 1, 2])
        assert_sample(estimate)

        estimate = self.general_estimator(operator(), state, np.array([0, 1, 2]))
        assert_sample(estimate)

    def test_concurrent_parametric_estimate(self) -> None:
        circuit = ParametricQuantumCircuit(n_qubits)
        circuit.add_X_gate(0)
        circuit.add_ParametricRX_gate(0)
        circuit.add_ParametricRY_gate(1)
        circuit.add_ParametricRZ_gate(2)

        state = ParametricCircuitQuantumState(n_qubits, circuit)

        estimates = self.general_estimator(operator(), state, [[0, 1, 2], [4, 5, 6]])
        estimate_list = list(estimates)
        assert len(estimate_list) == 2
        assert_sample(estimate_list[0])
        assert_sample(estimate_list[1])

        estimates = self.general_estimator(
            operator(), state, np.array([[0, 1, 2], [4, 5, 6]])
        )
        estimate_list = list(estimates)
        assert len(estimate_list) == 2
        assert_sample(estimate_list[0])
        assert_sample(estimate_list[1])
