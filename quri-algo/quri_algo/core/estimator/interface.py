# Licensed under the MIT License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      https://mit-license.org/
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from abc import abstractmethod
from typing import Any, Protocol, TypeVar, Union

from quri_parts.core.estimator import Estimate
from quri_parts.core.state import CircuitQuantumState, QuantumStateVector
from typing_extensions import TypeAlias

State: TypeAlias = Union[CircuitQuantumState, QuantumStateVector]
StateT = TypeVar("StateT", bound=State, contravariant=True)


class ExpectationValueEstimator(Protocol[StateT]):
    r"""Interface for estimator that computes the expectaion value of a function
    of an operator with some parameter. It can be understood as an object that
    computes:

    .. math::
        \langle f(O; \vec{\theta}) \rangle
    """

    @abstractmethod
    def __call__(self, state: StateT, *args: Any, **kwargs: Any) -> Estimate[complex]:
        """Computes the expectation value."""
        ...


class OperatorPowerEstimatorBase(ExpectationValueEstimator[StateT], Protocol):
    r"""Base class for any estimator that estimates the expectation value
    :math:`\langle U^k \rangle`, where :math:`U` is a unitary operator and `k`
    is the power.
    """

    @abstractmethod
    def __call__(
        self, state: StateT, operator_power: int | float, *args: Any, **kwargs: Any
    ) -> Estimate[complex]:
        ...
