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

from quri_parts.circuit import NonParametricQuantumCircuit
from quri_parts.circuit.noise import NoiseModel
from quri_parts.core.estimator import Estimatable, Estimate, QuantumEstimator, _StateT
from quri_parts.core.sampling import MeasurementCounts, Sampler
from quri_parts.qulacs import QulacsStateT
from quri_parts.qulacs.estimator import (
    create_qulacs_density_matrix_estimator,
    create_qulacs_vector_estimator,
)
from quri_parts.qulacs.sampler import (
    create_qulacs_noisesimulator_sampler,
    create_qulacs_vector_sampler,
)


class LogicalCircuitSimulator(ABC):
    """Abstract base class for logical circuit simulator.

    The primary objective is to define the interfaces to give VMs the ability
    to execute logical circuits.
    """

    @abstractmethod
    def sample(
        self,
        circuit: NonParametricQuantumCircuit,
        shots: int,
        noise_model: NoiseModel | None = None,
    ) -> MeasurementCounts: ...

    @abstractmethod
    def estimate(
        self,
        estimatable: Estimatable,
        state: _StateT,
        noise_model: NoiseModel | None = None,
    ) -> Estimate[complex]: ...


class QulacsSimulator(LogicalCircuitSimulator):
    """LogicalCircuitSimulator implementation using Qulacs."""

    def sample(
        self,
        circuit: NonParametricQuantumCircuit,
        shots: int,
        noise_model: NoiseModel | None = None,
    ) -> MeasurementCounts:
        return self._get_sampler(noise_model)(circuit, shots)

    def _get_sampler(self, noise_model: NoiseModel | None = None) -> Sampler:
        if noise_model is not None:
            return create_qulacs_noisesimulator_sampler(noise_model)
        return create_qulacs_vector_sampler()

    def estimate(
        self,
        estimatable: Estimatable,
        state: QulacsStateT,
        noise_model: NoiseModel | None = None,
    ) -> Estimate[complex]:
        return self._get_estimater(noise_model)(estimatable, state)

    def _get_estimater(
        self,
        noise_model: NoiseModel | None = None,
    ) -> QuantumEstimator[QulacsStateT]:
        if noise_model is not None:
            return create_qulacs_density_matrix_estimator(noise_model)
        return create_qulacs_vector_estimator()


# class CuQuantumSimulator(LogicalCircuitSimulator): ...
