# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from abc import abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING, Callable, Optional, Protocol

from typing_extensions import TypeAlias

if TYPE_CHECKING:
    import numpy as np  # noqa: F401
    import numpy.typing as npt  # noqa: F401

#: Represents a parameter vector subject to optimization.
#: (A gradient vector is also represented as Params.)
Params: TypeAlias = "npt.NDArray[np.float64]"

#: Cost function for optimization.
CostFunction: TypeAlias = Callable[[Params], float]

#: Gradient function for optimization.
GradientFunction: TypeAlias = Callable[[Params], Params]


class OptimizerStatus(Enum):
    """Status of optimization."""

    #: No error, not converged yet.
    SUCCESS = auto()
    #: The optimization failed and cannot be continued.
    FAILED = auto()
    #: The optimization converged.
    CONVERGED = auto()


@dataclass(frozen=True)
class OptimizerState:
    """An immutable (frozen) dataclass representing an optimizer state."""

    #: Current parameter values.
    params: Params
    #: Current value of the cost function.
    cost: float = 0.0
    #: Optimization status.
    status: OptimizerStatus = OptimizerStatus.SUCCESS
    #: Number of iterations.
    niter: int = 0
    #: Number of cost function calls.
    funcalls: int = 0
    #: Number of gradient function calls.
    gradcalls: int = 0

    @property
    def n_params(self) -> int:
        """Number of parameters."""
        return len(self.params)


class Optimizer(Protocol):
    """A protocol class for optimizers."""

    @abstractmethod
    def get_init_state(self, init_params: Params) -> OptimizerState:
        """Returns an initial state for optimization."""
        ...

    @abstractmethod
    def step(
        self,
        state: OptimizerState,
        cost_function: CostFunction,
        grad_function: Optional[GradientFunction] = None,
    ) -> OptimizerState:
        """Run a single optimization step and returns a new state."""
        ...
