# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass
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


@dataclass(frozen=True)
class OptimizerStateSPSA(OptimizerState):
    rng: np.random.Generator = np.random.default_rng()


class SPSA(Optimizer):
    r"""Simultaneous perturbation stochastic approximation (SPSA) optimizer. The
    implementation is heavily inspired by [1]. Given the parameters
    :math:`\theta_k` at an iteration :math:`k`, the updated parameters
    :math:`\theta_{k+1}` is given by

    .. math::
        \theta_{k+1} = \theta_k - \alpha_k g_k(\theta_k),
        g_k(\theta_k) = \frac{f(\theta_k+c_k \Delta_k)-f(\theta_k-c_k \Delta_k)}{2 c_k}
            \Delta_k^{-1},
        a_k = a / (A + k + 1)^\alpha, c_k = c / (k + 1)^\gamma,

    where :math:`f` is the cost function to be minimized, and :math:`\Delta_k`
    is a vector generated randomly. :math:`\Delta_k^{-1}` is defined as the
    element-wise inverse of :math:`\Delta_k`.
    The dimension of :math:`\Delta_k` is the same as that of :math:`\theta_k`.
    In this optimizer, :math:`\Delta_k` is generated from a Bernoulli distribution.
    Note that :math:`g_k(\theta_k)` works as an estimate of the first order
    gradient of the cost function :math:`f(\theta_k)`.

    The main advantage of the SPSA optimizer is that it requires only 2
    function evaluations to estimate the gradient in each iteration. Whereas
    the standard gradient-based optimizers require :math:`2p` or more
    function evaluations (:math:`p`: the parameter length/size) in each
    iteration to compute the gradient. Hence the SPSA could be useful when
    performing VQE with sampling.

    Args:
        a: :math:`a` in the parameter update rule that is defined above.
        c: :math:`c` in the parameter update rule that is defined above.
        alpha: :math:`\alpha` in the parameter update rule that is defined above.
        gamma: :math:`\gamma` in the parameter update rule that is defined above.
        A: :math:`A` in the parameter update rule that is defined above.
            A recommended choice is `A` = (10 or 100) multiplied by
            the maximum number of iterations that the optimizer runs.
        ftol: If not None, judge convergence by cost function tolerance.
            See :func:`~.tolerance.ftol` for details.

    Ref:
        [1]: J. C. Spall, "Implementation of the simultaneous perturbation algorithm
            for stochastic optimization,"
            in IEEE Transactions on Aerospace and Electronic Systems, vol. 34,
            no. 3, pp. 817-823, July 1998, doi: 10.1109/7.705889.
    """

    def __init__(
        self,
        a: float = 2.0 * np.pi / 10.0,
        c: float = 0.1,
        alpha: float = 0.602,
        gamma: float = 0.101,
        A: float = 0.0,
        ftol: Optional[float] = 1e-5,
        rng_seed: Optional[int] = None,
    ):
        if not 0.0 < a:
            raise ValueError(f"'a' must be larger than zero, But got {a}")
        if not 0.0 < alpha:
            raise ValueError(f"'alpha' must be larger than zero. But got {alpha}")
        if not 0.0 < gamma:  # NOTE: is this if statement really neccesary?
            raise ValueError(f"'gamma' must be larger than zero, But got {gamma}")

        self._a = a
        self._c = c
        self._alpha = alpha
        self._gamma = gamma
        self._A = A

        self._rng_seed = rng_seed

        self._ftol: Optional[Callable[[float, float], bool]] = None
        if ftol is not None:
            if not 0.0 < ftol:
                raise ValueError("ftol must be a positive float.")
            self._ftol = create_ftol(ftol)

    def get_init_state(self, init_params: Params) -> OptimizerStateSPSA:
        params = readonly_array(np.array(init_params, dtype=float))
        rng = np.random.default_rng(seed=self._rng_seed)
        return OptimizerStateSPSA(
            params=params,
            rng=rng,
        )

    def step(
        self,
        state: OptimizerState,
        cost_function: CostFunction,
        grad_function: Optional[GradientFunction] = None,
    ) -> OptimizerStateSPSA:
        if not isinstance(state, OptimizerStateSPSA):
            raise ValueError('state must have type "OptimizerStateSPSA".')

        rng = state.rng

        funcalls = state.funcalls
        gradcalls = state.gradcalls
        niter = state.niter + 1

        params = state.params.copy()

        if niter == 1:
            cost_prev = cost_function(params)
            funcalls += 1
        else:
            cost_prev = state.cost

        delta = 2 * rng.integers(2, size=state.n_params) - 1

        ck = self._c / (state.niter + 1) ** self._gamma
        params_plus = params + delta * ck
        params_minus = params - delta * ck

        # estimate of the first order gradient
        g = (
            (cost_function(params_plus) - cost_function(params_minus))
            / (2 * ck)
            / delta
        )
        funcalls += 2

        ak = self._a / (state.niter + self._A + 1) ** self._alpha
        params -= ak * g
        cost = cost_function(params)
        funcalls += 1

        if self._ftol and self._ftol(cost, cost_prev):
            status = OptimizerStatus.CONVERGED
        else:
            status = OptimizerStatus.SUCCESS

        return OptimizerStateSPSA(
            params=readonly_array(params),
            cost=cost,
            status=status,
            niter=niter,
            funcalls=funcalls,
            gradcalls=gradcalls,
            rng=rng,
        )
