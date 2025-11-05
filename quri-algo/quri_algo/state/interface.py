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
from typing import Any, Protocol, runtime_checkable

from quri_parts.core.state import QuantumState


@runtime_checkable
class StateFactory(Protocol):
    """Base class for state preparation."""

    @abstractmethod
    def __call__(self, *args: Any, **kwargs: Any) -> QuantumState:
        ...


@runtime_checkable
class TimeEvolutionStateFactory(StateFactory, Protocol):
    """Base class for state preparation that relies on Hamiltonian
    simulation."""

    @abstractmethod
    def __call__(
        self, evolution_time: float, *args: Any, **kwargs: Any
    ) -> QuantumState:
        ...
