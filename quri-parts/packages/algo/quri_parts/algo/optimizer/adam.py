# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from quri_parts.core.utils.array import readonly_array

from .interface import (
    CostFunction,
    GradientFunction,
    Optimizer,
    OptimizerState,
    OptimizerStatus,
    Params,
)
from .tolerance import ftol as create_ftol


def _const_zero_array_factory() -> Params:
    return readonly_array(np.zeros(0, dtype=float))


@dataclass(frozen=True)
class OptimizerStateAdam(OptimizerState):
    """Optimizer state for Adam."""

    m: Params = field(default_factory=_const_zero_array_factory)
    v: Params = field(default_factory=_const_zero_array_factory)


def _adam_grad_square_moving_average(
    v_prev: Params, grad: Params, m: Params, beta2: float, eps: float
) -> Params:
    return beta2 * v_prev + (1 - beta2) * grad**2


class Adam(Optimizer):
    """Adam optimization algorithm proposed in [1].

    Args:
        lr: learning rate.
        betas: coefficients used in the update rules of
            the moving averages of the gradient and its magnitude.
            ``betas`` represents the robustness of the optimizer. Hence,
            when using sampling, the higher values for ``betas``
            are recommended.
        eps: a small scaler number used for avoiding zero division.
        ftol: If not None, judge convergence by cost function tolerance.
            See :func:`~.tolerance.ftol` for details.

    Ref:
        [1] Adam: A Method for Stochastic Optimization,
        Diederik P. Kingma, Jimmy Ba (2014).
        https://arxiv.org/abs/1412.6980.
    """

    _grad_square_moving_average = _adam_grad_square_moving_average

    def __init__(
        self,
        lr: float = 0.05,
        betas: Sequence[float] = (0.9, 0.999),
        eps: float = 1e-9,
        ftol: Optional[float] = 1e-5,
    ):
        if not 0.0 < lr:
            raise ValueError(f"Learning rate must be larger than zero. But got {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(
                f"betas[0] must satisty 0.0 <= betas[0] < 1.0. But got {betas[0]}"
            )
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(
                f"betas[1] must satisty 0.0 <= betas[1] < 1.0. But got {betas[1]}"
            )
        if not 0.0 <= eps:
            raise ValueError(f"eps must NOT be smaller than zero. But got {eps}")
        self._lr = lr
        self._betas = betas
        self._eps = eps

        self._ftol: Optional[Callable[[float, float], bool]] = None
        if ftol is not None:
            if not 0.0 < ftol:
                raise ValueError("ftol must be a positive float.")
            self._ftol = create_ftol(ftol)

    def get_init_state(self, init_params: Params) -> OptimizerStateAdam:
        # Defensively copy init_params by np.array
        params = readonly_array(np.array(init_params, dtype=float))
        zeros = readonly_array(np.zeros(len(params), dtype=float))
        return OptimizerStateAdam(
            params=params,
            m=zeros,
            v=zeros,
        )

    def step(
        self,
        state: OptimizerState,
        cost_function: CostFunction,
        grad_function: Optional[GradientFunction] = None,
    ) -> OptimizerStateAdam:
        if not isinstance(state, OptimizerStateAdam):
            raise ValueError("state must have type OptimizerStateAdam.")

        if grad_function is None:
            raise ValueError("Adam requires a gradient function.")

        funcalls = state.funcalls
        gradcalls = state.gradcalls
        niter = state.niter + 1

        if niter == 1:
            cost_prev = cost_function(state.params)
            funcalls += 1
        else:
            cost_prev = state.cost

        beta1, beta2 = self._betas

        grad: Params = grad_function(state.params)
        gradcalls += 1

        m = beta1 * state.m + (1 - beta1) * grad
        v = self.__class__._grad_square_moving_average(
            state.v, grad, m, beta2, self._eps
        )
        m_hat = m / (1 - beta1**niter)
        v_hat = v / (1 - beta2**niter)
        params = state.params - self._lr * m_hat / (np.sqrt(v_hat) + self._eps)

        cost = cost_function(params)
        funcalls += 1

        if self._ftol and self._ftol(cost, cost_prev):
            status = OptimizerStatus.CONVERGED
        else:
            status = OptimizerStatus.SUCCESS

        return OptimizerStateAdam(
            params=readonly_array(params),
            cost=cost,
            status=status,
            niter=niter,
            funcalls=funcalls,
            gradcalls=gradcalls,
            m=readonly_array(m),
            v=readonly_array(v),
        )


def _adabelief_grad_square_moving_average(
    v_prev: Params, grad: Params, m: Params, beta2: float, eps: float
) -> Params:
    return beta2 * v_prev + (1 - beta2) * (grad - m) ** 2 + eps


class AdaBelief(Adam):
    """AdaBelief optimization algorithm proposed in [1].

    Args:
        lr: learning rate.
        betas: coefficients used in the update rules of
            the moving averages of the gradient and its magnitude.
            ``betas`` represents the robustness of the optimizer. Hence,
            when using sampling, the higher values for ``betas``
            are recommended.
        eps: a small scaler number used for avoiding zero division.
        ftol: If not None, judge convergence by cost function tolerance.
            See :func:`~.tolerance.ftol` for details.

    Ref:
        [1] AdaBelief Optimizer: Adapting Stepsizes by the Belief in Observed Gradients,
        Juntang Zhuang, Tommy Tang, Yifan Ding, Sekhar Tatikonda, Nicha Dvornek,
        Xenophon Papademetris, James S. Duncan (2020).
        https://arxiv.org/abs/2010.07468.
    """

    _grad_square_moving_average = _adabelief_grad_square_moving_average

    def __init__(
        self,
        lr: float = 1e-3,
        betas: Sequence[float] = (0.9, 0.99),
        eps: float = 1e-16,
        ftol: Optional[float] = 1e-5,
    ):
        super().__init__(lr, betas, eps, ftol)
