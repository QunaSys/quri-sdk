# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import Counter

import numpy as np
import pytest

from quri_parts.core.sampling import (
    ideal_sample_from_state_vector,
    sample_from_state_vector,
)


class TestSampleFromStateVector:
    def test_sample_from_state_vector(self) -> None:
        n_qubits = 2
        for i in range(2**n_qubits):
            phase = np.random.random()
            state = np.zeros(2**n_qubits, dtype=np.complex128)
            state[i] = np.exp(1j * phase)
            assert sample_from_state_vector(state, 1000) == Counter({i: 1000})

    def test_invalid_input(self) -> None:
        with pytest.raises(
            AssertionError, match="Length of the state vector must be a power of 2."
        ):
            sample_from_state_vector(
                np.array([0.5, 0.1, np.sqrt(1 - 0.25 - 0.01)]), 1000
            )

        with pytest.raises(ValueError, match="probabilities do not sum to 1"):
            sample_from_state_vector(
                np.random.random(4) + 1j * np.random.random(4), 1000
            )


class TestIdealSampleFromStateVector:
    def test_ideal_sample_from_state_vector(self) -> None:
        n_qubits = 2
        state_vector = np.array(
            [
                0.13106223 + 0.70435299j,
                0.16605566 - 0.36973591j,
                0.10202236 + 0.48950168j,
                0.18068102 - 0.19940998j,
            ]
        )
        sampled_cnt = ideal_sample_from_state_vector(state_vector, 1000)
        expected_cnt = Counter(
            {0: 513.29044785, 1: 164.27912771, 2: 250.02045389, 3: 72.40997054}
        )

        for i in range(2**n_qubits):
            assert np.isclose(sampled_cnt[i], expected_cnt[i])
        assert np.isclose(sum(expected_cnt.values()), 1000)

    def test_invalid_input(self) -> None:
        with pytest.raises(
            AssertionError, match="Length of the state vector must be a power of 2."
        ):
            ideal_sample_from_state_vector(
                np.array([0.5, 0.1, np.sqrt(1 - 0.25 - 0.01)]), 1000
            )

        with pytest.raises(ValueError, match="probabilities do not sum to 1"):
            ideal_sample_from_state_vector(
                np.random.random(4) + 1j * np.random.random(4), 1000
            )
