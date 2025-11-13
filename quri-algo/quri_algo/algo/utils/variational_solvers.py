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
from typing import Any, Sequence

from quri_parts.algo.optimizer import (
    CostFunction,
    GradientFunction,
    Optimizer,
    OptimizerState,
    OptimizerStatus,
    Params,
)
from quri_parts.algo.optimizer.lbfgs import LBFGS
from scipy.optimize import OptimizeResult, minimize


class VariationalSolver(ABC):
    @abstractmethod
    def __call__(
        self,
        cost: CostFunction,
        grad: GradientFunction | None,
        init_params: Params,
        *args: Any,
        **kwds: Any
    ) -> Sequence[OptimizerState]:
        ...


class QURIPartsVariationalSolver(VariationalSolver):
    def __init__(self, optimizer: Optimizer):
        self.optimizer = optimizer

    def __call__(
        self, cost: CostFunction, grad: GradientFunction | None, init_params: Params
    ) -> Sequence[OptimizerState]:
        optimizer_state = self.optimizer.get_init_state(init_params)
        optimizer_history = []
        while True:
            optimizer_state = self.optimizer.step(optimizer_state, cost, grad)
            optimizer_history.append(optimizer_state)
            if optimizer_state.status in [
                OptimizerStatus.CONVERGED,
                OptimizerStatus.FAILED,
            ]:
                break
        return optimizer_history


class SciPyVariationalSolver(VariationalSolver):
    def __call__(
        self,
        cost: CostFunction,
        grad: GradientFunction | None,
        init_params: Params,
        **scipy_optimization_options: Any
    ) -> Sequence[OptimizerState]:
        optimizer_history = []

        def callback(intermediate_result: OptimizeResult) -> None:
            status = (
                OptimizerStatus.CONVERGED
                if intermediate_result.success
                else OptimizerStatus.FAILED
            )
            optimizer_state = OptimizerState(
                intermediate_result.x,
                intermediate_result.fun,
                status,
                intermediate_result.nit,
                intermediate_result.nfev,
                intermediate_result.nhev,
            )
            optimizer_history.append(optimizer_state)

        res = minimize(
            cost, init_params, jac=grad, callback=callback, **scipy_optimization_options
        )
        status = OptimizerStatus.CONVERGED if res.success else OptimizerStatus.FAILED
        optimizer_history.append(
            OptimizerState(res.x, res.fun, status, res.nit, res.nfev, res.nhev)
        )

        return optimizer_history


def create_default_variational_solver() -> VariationalSolver:
    """Create a default variational solver.

    Returns:
        A variational solver instance.
    """
    return QURIPartsVariationalSolver(LBFGS())
