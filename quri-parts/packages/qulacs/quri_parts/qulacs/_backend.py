# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from abc import ABC, abstractmethod
from typing import cast

import numpy as np
import qulacs as ql
from numpy import complex128
from numpy.typing import NDArray

from quri_parts.qulacs.utils import cast_to_list


def _get_qubit_count(state_vector: NDArray[complex128]) -> int:
    n_qubits = np.log2(state_vector.shape[0])
    if not n_qubits.is_integer():
        raise ValueError(
            f"Length of state_vector ({state_vector.shape[0]}) "
            "must be a power of 2"
        )
    return int(n_qubits)


class QulacsBackend(ABC):
    """Abstract base class for qulacs backends.

    Subclasses should either:
    - Inherit from :class:`DefaultQulacsBackend` to reuse default implementations
      (recommended for backends that share most behavior with the default backend)
    - Inherit from this class directly to provide a fully custom implementation
    """

    @abstractmethod
    def init_state(
        self, qubit_count: int, init_state: NDArray[complex128]
    ) -> ql.QuantumState:
        """Create a qulacs QuantumState loaded with ``init_state``."""
        ...

    @abstractmethod
    def init_zero_state(self, qubit_count: int) -> ql.QuantumState:
        """Create a qulacs QuantumState initialised to |0⟩.

        MPI-aware backends should override this to avoid the O(2**n)
        allocation that would otherwise happen on every rank.
        """
        ...

    @abstractmethod
    def init_density_matrix(
        self, qubit_count: int, init_state: NDArray[complex128]
    ) -> ql.DensityMatrix:
        """Create a qulacs DensityMatrix loaded with ``init_state``."""
        ...

    @abstractmethod
    def init_noise_simulator(
        self, qs_circuit: ql.QuantumCircuit, qs_state: ql.QuantumState
    ) -> ql.NoiseSimulator:
        """Create a qulacs NoiseSimulator for ``qs_circuit`` and ``qs_state``."""
        ...

    @abstractmethod
    def should_use_multinomial(self, n_shots: int, qubit_count: int) -> bool:
        """Return True when multinomial sampling is preferable over qulacs sampling."""
        ...

    @abstractmethod
    def get_state_vector(
        self, state: ql.QuantumState, qubit_count: int
    ) -> NDArray[complex128]:
        """Return the state vector held by ``state``.

        Backends may override this to reconstruct the vector from a
        non-standard internal representation before returning it.
        """
        ...

    @abstractmethod
    def get_marginal_probability(
        self,
        state_vector: NDArray[complex128],
        measured_values: dict[int, int],
    ) -> float:
        """Compute the marginal probability of measuring ``measured_values``."""
        ...


class DefaultQulacsBackend(QulacsBackend):
    def init_state(
        self, qubit_count: int, init_state: NDArray[complex128]
    ) -> ql.QuantumState:
        if init_state.ndim != 1:
            raise ValueError(
                f"init_state must be a 1D array, got {init_state.ndim}D"
            )
        if len(init_state) != 2**qubit_count:
            raise ValueError(
                f"Length of init_state ({len(init_state)}) does not match "
                f"2**qubit_count ({2**qubit_count})"
            )

        qulacs_state = ql.QuantumState(qubit_count)
        qulacs_state.load(cast_to_list(init_state))
        return qulacs_state

    def init_zero_state(self, qubit_count: int) -> ql.QuantumState:
        return ql.QuantumState(qubit_count)

    def init_density_matrix(
        self, qubit_count: int, init_state: NDArray[complex128]
    ) -> ql.DensityMatrix:
        expected = 2**qubit_count
        if init_state.ndim == 1:
            if len(init_state) != expected:
                raise ValueError(
                    f"Length of init_state ({len(init_state)}) does not match "
                    f"2**qubit_count ({expected})"
                )
        elif init_state.ndim == 2:
            if init_state.shape != (expected, expected):
                raise ValueError(
                    f"Shape of init_state {init_state.shape} does not match "
                    f"({expected}, {expected})"
                )
        else:
            raise ValueError(
                f"init_state must be 1D or 2D, got {init_state.ndim}D"
            )
        density_matrix = ql.DensityMatrix(qubit_count)
        density_matrix.load(init_state)
        return density_matrix

    def init_noise_simulator(
        self, qs_circuit: ql.QuantumCircuit, qs_state: ql.QuantumState
    ) -> ql.NoiseSimulator:
        return ql.NoiseSimulator(qs_circuit, qs_state)

    def get_state_vector(
        self, state: ql.QuantumState, qubit_count: int
    ) -> NDArray[complex128]:
        # cast required due to incomplete qulacs type stubs
        # https://github.com/qulacs/qulacs/issues/537
        return cast(NDArray[complex128], state.get_vector())

    def should_use_multinomial(self, n_shots: int, qubit_count: int) -> bool:
        return n_shots > int(2 ** max(int(qubit_count), 10))

    def get_marginal_probability(
        self,
        state_vector: NDArray[complex128],
        measured_values: dict[int, int],
    ) -> float:
        n_qubits = _get_qubit_count(state_vector)
        if measured_values and max(measured_values.keys()) >= n_qubits:
            raise ValueError(
                f"The specified qubit index {max(measured_values.keys())} "
                f"is out of range (n_qubits={n_qubits})."
            )
        qulacs_state = self.init_state(n_qubits, state_vector)
        measured = [measured_values.get(i, 2) for i in range(n_qubits)]
        return cast(float, qulacs_state.get_marginal_probability(measured))


DEFAULT_BACKEND = DefaultQulacsBackend()
