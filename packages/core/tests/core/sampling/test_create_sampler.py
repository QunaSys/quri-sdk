# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
from collections.abc import Iterable, Sequence
from typing import Union
from unittest import TestCase, mock

import numpy as np
import pytest

from quri_parts.circuit import (
    CONST,
    LinearMappedUnboundParametricQuantumCircuit,
    NonParametricQuantumCircuit,
    QuantumCircuit,
    UnboundParametricQuantumCircuit,
)
from quri_parts.core.sampling import (
    ConcurrentParametricStateSampler,
    GeneralSampler,
    MeasurementCounts,
    create_concurrent_parametric_sampler_from_concurrent_sampler,
    create_concurrent_parametric_state_sampler_from_concurrent_state_sampler,
    create_concurrent_sampler_from_sampling_backend,
    create_parametric_sampler_from_sampler,
    create_parametric_state_sampler_from_state_sampler,
    create_sampler_from_sampling_backend,
    sample_from_probability_distribution,
)
from quri_parts.core.state import (
    CircuitQuantumState,
    ParametricCircuitQuantumState,
    ParametricQuantumStateVector,
    QuantumStateVector,
    quantum_state,
)


def test_sample_from_probibility_distribution() -> None:
    norm = 1.0
    p1 = 0.4
    prob = np.array([p1, norm - p1])
    cnts = sample_from_probability_distribution(1000, prob)
    assert sum(list(cnts.values())) == 1000

    norm = 1.000000000002
    p1 = 0.4
    prob = np.array([p1, norm - p1])
    cnts = sample_from_probability_distribution(1000, prob)
    assert sum(list(cnts.values())) == 1000

    norm = 1.001
    p1 = 0.4
    prob = np.array([p1, norm - p1])
    with pytest.raises(AssertionError, match="Probabilty does not sum to 1.0"):
        cnts = sample_from_probability_distribution(1000, prob)

    # Potentially dangerous case: large amount of small probabilities (p < 1e-12)
    # When they are rounded away, sum of the rest of the probabilities slightly > 1.
    n = 2**12
    n_small = 2000
    small_prob = -np.ones(n_small) * 1e-14
    big_prob = np.ones(n - n_small) * (1 - np.sum(small_prob)) / (n - n_small)
    prob = np.hstack([big_prob, small_prob])
    with pytest.raises(ValueError):
        rng = np.random.default_rng()
        rng.multinomial(1000, prob.round(12))
    cnts = sample_from_probability_distribution(1000, prob)
    assert sum(list(cnts.values())) == 1000


def fake_sampler(circuit: NonParametricQuantumCircuit, shot: int) -> MeasurementCounts:
    cnt = 0.0
    for g in circuit.gates:
        cnt += sum(g.params)
    return {0: int(cnt) * shot}


def fake_concurrent_sampler(
    circuit_shot_pairs: Iterable[tuple[NonParametricQuantumCircuit, int]]
) -> Iterable[MeasurementCounts]:
    return [fake_sampler(c, s) for c, s in circuit_shot_pairs]


def fake_state_sampler(
    state: Union[CircuitQuantumState, QuantumStateVector], shot: int
) -> MeasurementCounts:
    cnt = 0.0
    for g in state.circuit.gates:
        cnt += sum(g.params)
    if isinstance(state, QuantumStateVector):
        cnt *= int(np.linalg.norm(state.vector, 1))
    return {0: int(cnt) * shot}


def fake_concurrent_state_sampler(
    state_shot_pairs: Iterable[
        tuple[Union[CircuitQuantumState, QuantumStateVector], int]
    ]
) -> Iterable[MeasurementCounts]:
    return [fake_state_sampler(c, s) for c, s in state_shot_pairs]


class TestParametricSampler(TestCase):
    param_circuit_1: UnboundParametricQuantumCircuit
    param_circuit_2: LinearMappedUnboundParametricQuantumCircuit
    param_state_1: ParametricCircuitQuantumState
    param_state_2: ParametricQuantumStateVector

    def setUp(self) -> None:
        self.param_circuit_1 = UnboundParametricQuantumCircuit(2)
        self.param_circuit_1.add_ParametricRX_gate(0)
        self.param_circuit_1.add_ParametricRZ_gate(1)

        self.param_circuit_2 = LinearMappedUnboundParametricQuantumCircuit(2)
        a, b = self.param_circuit_2.add_parameters("a", "b")
        self.param_circuit_2.add_ParametricRX_gate(0, {a: -1, b: -2, CONST: -3})
        self.param_circuit_2.add_ParametricRZ_gate(1, {a: 4, b: 5, CONST: 6})

        self.param_state_1 = quantum_state(2, circuit=self.param_circuit_1)
        self.param_state_2 = quantum_state(
            2, circuit=self.param_circuit_2, vector=np.ones(4) / 2
        )

    def test_create_parametric_sampler_from_sampler(self) -> None:
        sampler = fake_sampler
        param_sampler = create_parametric_sampler_from_sampler(sampler)

        assert param_sampler(self.param_circuit_1, 1000, [1, 2]) == {0: 3000}
        assert param_sampler(self.param_circuit_1, 2000, [3, 4]) == {0: 14000}

        assert param_sampler(self.param_circuit_2, 1000, [1, 2]) == {
            0: (-8 + 20) * 1000
        }
        assert param_sampler(self.param_circuit_2, 2000, [3, 4]) == {
            0: (-14 + 38) * 2000
        }

    def test_create_concurrent_parametric_sampler_from_concurrent_sampler(self) -> None:
        concurrent_sampler = fake_concurrent_sampler
        concurrent_param_sampler = (
            create_concurrent_parametric_sampler_from_concurrent_sampler(
                concurrent_sampler
            )
        )

        assert concurrent_param_sampler(
            self.param_circuit_1, [(1000, [1, 2]), (2000, [3, 4])]
        ) == [{0: 3000}, {0: 14000}]

        assert concurrent_param_sampler(
            self.param_circuit_2,
            [(1000, [1, 2]), (2000, [3, 4])],
        ) == [{0: (-8 + 20) * 1000}, {0: (-14 + 38) * 2000}]

    def test_create_parametric_state_sampler_from_state_sampler(self) -> None:
        state_sampler = fake_state_sampler
        parametric_state_sampler = create_parametric_state_sampler_from_state_sampler(
            state_sampler
        )
        assert parametric_state_sampler(self.param_state_1, 1000, [1, 2]) == {0: 3000}
        assert parametric_state_sampler(self.param_state_1, 2000, [3, 4]) == {0: 14000}
        assert parametric_state_sampler(self.param_state_2, 1000, [1, 2]) == {
            0: (-8 + 20) * 1000 * 2
        }
        assert parametric_state_sampler(self.param_state_2, 2000, [3, 4]) == {
            0: (-14 + 38) * 2000 * 2
        }

    def test_create_concurrent_parametric_state_sampler_from_concurrent_state_sampler(
        self,
    ) -> None:
        concurrent_state_sampler = fake_concurrent_state_sampler
        concurrent_parametric_state_sampler: ConcurrentParametricStateSampler[
            Union[ParametricCircuitQuantumState, ParametricQuantumStateVector]
        ] = create_concurrent_parametric_state_sampler_from_concurrent_state_sampler(
            concurrent_state_sampler
        )
        assert concurrent_parametric_state_sampler(
            self.param_state_1, [(1000, [1, 2]), (2000, [3, 4])]
        ) == [{0: 3000}, {0: 14000}]

        assert concurrent_parametric_state_sampler(
            self.param_state_2, [(1000, [1, 2]), (2000, [3, 4])]
        ) == [{0: (-8 + 20) * 1000 * 2}, {0: (-14 + 38) * 2000 * 2}]


class TestGeneralSampler(TestCase):
    param_circuit_1: UnboundParametricQuantumCircuit
    param_circuit_2: LinearMappedUnboundParametricQuantumCircuit
    param_state_1: ParametricCircuitQuantumState
    param_state_2: ParametricQuantumStateVector
    general_sampler: GeneralSampler[
        Union[CircuitQuantumState, QuantumStateVector],
        Union[ParametricQuantumStateVector, ParametricQuantumStateVector],
    ]

    @classmethod
    def setUpClass(cls) -> None:
        cls.param_circuit_1 = UnboundParametricQuantumCircuit(2)
        cls.param_circuit_1.add_ParametricRX_gate(0)
        cls.param_circuit_1.add_ParametricRZ_gate(1)

        cls.param_circuit_2 = LinearMappedUnboundParametricQuantumCircuit(2)
        a, b = cls.param_circuit_2.add_parameters("a", "b")
        cls.param_circuit_2.add_ParametricRX_gate(0, {a: -1, b: -2, CONST: -3})
        cls.param_circuit_2.add_ParametricRZ_gate(1, {a: 4, b: 5, CONST: 6})

        cls.param_state_1 = quantum_state(2, circuit=cls.param_circuit_1)
        cls.param_state_2 = quantum_state(
            2, circuit=cls.param_circuit_2, vector=np.ones(4) / 2
        )

        cls.general_sampler = GeneralSampler(
            sampler=fake_sampler, state_sampler=fake_state_sampler
        )

    def test_sampler_input(self) -> None:
        circuit = self.param_circuit_1.bind_parameters([1, 2])
        assert self.general_sampler(circuit, 1000) == {0: 3000}
        circuit = self.param_circuit_1.bind_parameters([3, 4])
        assert self.general_sampler(circuit, 2000) == {0: 14000}
        circuit = self.param_circuit_2.bind_parameters([1, 2])
        assert self.general_sampler(circuit, 1000) == {0: (-8 + 20) * 1000}
        circuit = self.param_circuit_2.bind_parameters([3, 4])
        assert self.general_sampler(circuit, 2000) == {0: (-14 + 38) * 2000}

    def test_concurrent_sampler_input(self) -> None:
        circuit_1 = self.param_circuit_1.bind_parameters([1, 2])
        circuit_2 = self.param_circuit_1.bind_parameters([3, 4])
        assert self.general_sampler((circuit_1, 1000), (circuit_2, 2000)) == [
            {0: 3000},
            {0: 14000},
        ]

        assert self.general_sampler([(circuit_1, 1000), (circuit_2, 2000)]) == [
            {0: 3000},
            {0: 14000},
        ]

    def test_param_sampler_input(self) -> None:
        assert self.general_sampler(self.param_circuit_1, 1000, [1, 2]) == {0: 3000}
        assert self.general_sampler(self.param_circuit_1, 2000, [3, 4]) == {0: 14000}
        assert self.general_sampler(self.param_circuit_2, 1000, [1, 2]) == {
            0: (-8 + 20) * 1000
        }
        assert self.general_sampler(self.param_circuit_2, 2000, [3, 4]) == {
            0: (-14 + 38) * 2000
        }

    def test_concurrent_param_sampler_input(self) -> None:
        assert self.general_sampler(
            self.param_circuit_1, [(1000, [1, 2]), (2000, [3, 4])]
        ) == [{0: 3000}, {0: 14000}]

        assert self.general_sampler(
            self.param_circuit_2, [(1000, [1, 2]), (2000, [3, 4])]
        ) == [{0: (-8 + 20) * 1000}, {0: (-14 + 38) * 2000}]

        assert self.general_sampler(
            (self.param_circuit_1, 1000, [1, 2]),
            (self.param_circuit_1, 2000, [3, 4]),
            (self.param_circuit_2, 1000, [1, 2]),
            (self.param_circuit_2, 2000, [3, 4]),
        ) == [{0: 3000}, {0: 14000}, {0: (-8 + 20) * 1000}, {0: (-14 + 38) * 2000}]

        assert self.general_sampler(
            [
                (self.param_circuit_1, 1000, [1, 2]),
                (self.param_circuit_1, 2000, [3, 4]),
                (self.param_circuit_2, 1000, [1, 2]),
                (self.param_circuit_2, 2000, [3, 4]),
            ]
        ) == [{0: 3000}, {0: 14000}, {0: (-8 + 20) * 1000}, {0: (-14 + 38) * 2000}]

    def test_state_sampler_input(self) -> None:
        state = self.param_state_1.bind_parameters([1.0, 2.0])
        assert self.general_sampler(state, 1000) == {0: 3000}
        state = self.param_state_1.bind_parameters([3.0, 4.0])
        assert self.general_sampler(state, 2000) == {0: 14000}

        state_vector = self.param_state_2.bind_parameters([1.0, 2.0])
        assert self.general_sampler(state_vector, 1000) == {0: (-8 + 20) * 1000 * 2}
        state_vector = self.param_state_2.bind_parameters([3.0, 4.0])
        assert self.general_sampler(state_vector, 2000) == {0: (-14 + 38) * 2000 * 2}

    def test_concurrent_state_sampler_input(self) -> None:
        state_1 = self.param_state_1.bind_parameters([1.0, 2.0])
        state_2 = self.param_state_1.bind_parameters([3.0, 4.0])
        assert self.general_sampler((state_1, 1000), (state_2, 2000)) == [
            {0: 3000},
            {0: 14000},
        ]
        assert self.general_sampler([(state_1, 1000), (state_2, 2000)]) == [
            {0: 3000},
            {0: 14000},
        ]

    def test_param_state_sampler_input(self) -> None:
        assert self.general_sampler(self.param_state_1, 1000, [1.0, 2.0]) == {0: 3000}
        assert self.general_sampler(self.param_state_1, 2000, [3.0, 4.0]) == {0: 14000}
        assert self.general_sampler(self.param_state_2, 1000, [1.0, 2.0]) == {
            0: (-8 + 20) * 1000 * 2
        }
        assert self.general_sampler(self.param_state_2, 2000, [3.0, 4.0]) == {
            0: (-14 + 38) * 2000 * 2
        }

    def test_concurrent_param_state_sampler_input(self) -> None:
        assert self.general_sampler(
            self.param_state_1, [(1000, [1, 2]), (2000, [3, 4])]
        ) == [{0: 3000}, {0: 14000}]

        assert self.general_sampler(
            self.param_state_2, [(1000, [1, 2]), (2000, [3, 4])]
        ) == [{0: (-8 + 20) * 1000 * 2}, {0: (-14 + 38) * 2000 * 2}]

        assert self.general_sampler(
            (self.param_state_1, 1000, [1, 2]),
            (self.param_state_1, 2000, [3, 4]),
            (self.param_state_2, 1000, [1, 2]),
            (self.param_state_2, 2000, [3, 4]),
        ) == [
            {0: 3000},
            {0: 14000},
            {0: (-8 + 20) * 1000 * 2},
            {0: (-14 + 38) * 2000 * 2},
        ]

        assert self.general_sampler(
            [
                (self.param_state_1, 1000, [1, 2]),
                (self.param_state_1, 2000, [3, 4]),
                (self.param_state_2, 1000, [1, 2]),
                (self.param_state_2, 2000, [3, 4]),
            ]
        ) == [
            {0: 3000},
            {0: 14000},
            {0: (-8 + 20) * 1000 * 2},
            {0: (-14 + 38) * 2000 * 2},
        ]

    def test_mixed_concurrent_input(self) -> None:
        assert self.general_sampler(
            (self.param_circuit_1.bind_parameters([1, 2]), 1000),
            (self.param_circuit_1, 2000, [3, 4]),
            (self.param_state_2.bind_parameters([1, 2]), 1000),
            (self.param_state_2, 2000, [3, 4]),
        ) == [
            {0: 3000},
            {0: 14000},
            {0: (-8 + 20) * 1000 * 2},
            {0: (-14 + 38) * 2000 * 2},
        ]

        assert self.general_sampler(
            [
                (self.param_circuit_1.bind_parameters([1, 2]), 1000),
                (self.param_circuit_1, 2000, [3, 4]),
                (self.param_state_2.bind_parameters([1, 2]), 1000),
                (self.param_state_2, 2000, [3, 4]),
            ]
        ) == [
            {0: 3000},
            {0: 14000},
            {0: (-8 + 20) * 1000 * 2},
            {0: (-14 + 38) * 2000 * 2},
        ]

    def test_error_raises_correctly(self) -> None:
        # Test sample
        with pytest.raises(
            ValueError,
            match=(
                "Shot expected to be integer, but got <class 'float'>. "
                "Input value is 0.87."
            ),
        ):
            self.general_sampler(self.param_circuit_1.bind_parameters([1.0, 2.0]), 0.87)  # type: ignore # noqa: E501

        with pytest.raises(
            TypeError,
            match=re.escape("_sample() takes 3 positional arguments but 4 were given"),
        ):
            self.general_sampler(
                self.param_circuit_1.bind_parameters([1.0, 2.0]), 0.87, [3.0, 4.0]  # type: ignore # noqa: E501
            )

        # Test param sample
        with pytest.raises(TypeError):
            self.general_sampler(self.param_circuit_1, [1.0, 2.0], 100)  # type: ignore

        with pytest.raises(
            ValueError,
            match=re.escape(
                "Circuit parameter is expected to be an iterable or an array, "
                "but got <class 'int'>. Input value is 909090."
            ),
        ):
            self.general_sampler(self.param_circuit_1, 100, 909090)  # type: ignore

        with pytest.raises(
            TypeError,
            match="argument 'params': Can't extract `str` to `Vec`",
        ):
            self.general_sampler(self.param_circuit_1, 100, "ab")  # type: ignore

        # Test state sample
        with pytest.raises(
            ValueError,
            match=(
                "Shot expected to be integer, but got <class 'float'>. "
                "Input value is 0.87."
            ),
        ):
            self.general_sampler(self.param_state_1.bind_parameters([1.0, 2.0]), 0.87)  # type: ignore  # noqa: E501

        with pytest.raises(
            TypeError,
            match=re.escape(
                "_sample_state() takes 3 positional arguments but 4 were given"
            ),
        ):
            self.general_sampler(
                self.param_state_1.bind_parameters([1.0, 2.0]), 0.87, [3.0, 4.0]  # type: ignore    # noqa: E501
            )

        # Test param sample
        with pytest.raises(TypeError):
            self.general_sampler(self.param_state_1, [1.0, 2.0], 100)  # type: ignore

        with pytest.raises(
            ValueError,
            match=re.escape(
                "Circuit parameter is expected to be an iterable or an array, "
                "but got <class 'int'>. Input value is 909090."
            ),
        ):
            self.general_sampler(self.param_state_1, 100, 909090)  # type: ignore

        with pytest.raises(
            TypeError,
            match="argument 'params': Can't extract `str` to `Vec`",
        ):
            self.general_sampler(self.param_state_1, 100, "ab")  # type: ignore


def create_mock_backend(counts: Sequence[MeasurementCounts]) -> mock.Mock:
    def create_job(c: MeasurementCounts) -> mock.Mock:
        job = mock.Mock()
        result = job.result.return_value
        result.counts = c
        return job

    backend = mock.Mock()
    backend.sample.side_effect = [create_job(c) for c in counts]

    return backend


def test_create_sampler_from_sampling_backend() -> None:
    counts = {0: 100, 1: 200, 2: 300, 3: 400}
    backend = create_mock_backend([counts])

    sampler = create_sampler_from_sampling_backend(backend)
    circuit = QuantumCircuit(3)
    sampling_result = sampler(circuit, 1000)

    assert sampling_result == counts
    backend.sample.assert_called_with(circuit, 1000)


def test_create_concurrent_sampler_from_sampling_backend() -> None:
    counts = [
        {0: 100, 1: 200, 2: 300, 3: 400},
        {0: 300, 1: 300, 2: 200, 3: 100},
    ]
    backend = create_mock_backend(counts)

    sampler = create_concurrent_sampler_from_sampling_backend(backend)
    circuits = [QuantumCircuit(3), QuantumCircuit(2)]
    sampling_results = sampler([(circuits[0], 1000), (circuits[1], 900)])

    assert list(sampling_results) == counts
    assert backend.sample.call_args_list == [
        mock.call(circuits[0], 1000),
        mock.call(circuits[1], 900),
    ]
