# Licensed under the MIT License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      https://mit-license.org/
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from abc import ABC, abstractmethod
from typing import Any, NamedTuple, Protocol

import numpy as np
from quri_parts.circuit import NonParametricQuantumCircuit
from quri_parts.core.estimator import Estimate, QuantumEstimator
from quri_parts.core.state import CircuitQuantumState


class _Estimate(NamedTuple):
    value: float
    error: float = 0.0


class CostFunction(Protocol):
    estimator: QuantumEstimator[CircuitQuantumState]

    @abstractmethod
    def __call__(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Estimate[complex]:
        pass


class NonLocalCostFunction(CostFunction, ABC):
    @abstractmethod
    def __call__(
        self,
        target_circuit: NonParametricQuantumCircuit,
        trial_circuit: NonParametricQuantumCircuit,
        *args: Any,
        **kwargs: Any,
    ) -> Estimate[complex]:
        pass


class LocalCostFunction(CostFunction, ABC):
    restriction_size: int

    @abstractmethod
    def _local_estimate(
        self,
        target_circuit: NonParametricQuantumCircuit,
        trial_circuit: NonParametricQuantumCircuit,
        index: int,
        *args: Any,
        **kwargs: Any,
    ) -> Estimate[complex]:
        pass

    def __call__(
        self,
        target_circuit: NonParametricQuantumCircuit,
        trial_circuit: NonParametricQuantumCircuit,
        *args: Any,
        **kwargs: Any,
    ) -> Estimate[complex]:
        qubit_count = target_circuit.qubit_count
        estimates = [
            self._local_estimate(target_circuit, trial_circuit, i, *args, **kwargs)
            for i in range(qubit_count)
        ]
        cost = sum(e.value.real for e in estimates) / qubit_count
        err = np.sqrt(sum(e.error.real**2 for e in estimates)) / qubit_count

        return _Estimate(cost, err)
