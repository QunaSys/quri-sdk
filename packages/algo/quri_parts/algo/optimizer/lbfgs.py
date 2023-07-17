# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import warnings
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Callable, Optional, cast

import numpy as np

if TYPE_CHECKING:
    import numpy.typing as npt  # noqa: F401

from scipy.optimize.linesearch import (
    LineSearchWarning,
    line_search_wolfe1,
    line_search_wolfe2,
)

from quri_parts.core.utils.array import readonly_array

from .interface import (
    CostFunction,
    GradientFunction,
    Optimizer,
    OptimizerState,
    OptimizerStatus,
    Params,
)
from .tolerance import gtol as create_gtol


def _const_zero_array_factory() -> Params:
    return readonly_array(np.zeros(0, dtype=float))


@dataclass(frozen=True)
class OptimizerStateLBFGS(OptimizerState):
    """Optimizer state for LBFGS."""

    grad: Params = field(default_factory=_const_zero_array_factory)

    p: Params = field(default_factory=_const_zero_array_factory)
    s: Params = field(default_factory=_const_zero_array_factory)
    y: Params = field(default_factory=_const_zero_array_factory)
    rho: Params = field(default_factory=_const_zero_array_factory)

    cost_prev: float = 0.0
    ind: int = 0
    a: Params = field(default_factory=_const_zero_array_factory)


class LBFGS(Optimizer):
    r"""L-BFGS (Limited memory Bryden-Fletcher-Goldfarb-Shanno) optimizer.
    Partially inspired by SciPy implementation of BFGS optimizer [1]. For the
    details of algorithm, see [2].

    Args:
        c1: coefficient in strong Wolfe condition. It determines
            the range of the cost function.
        c2: coefficient in strong Wolfe condition. It determines
            the range of the derivative of the cost function.
        amin: lower bound of the value of the step size that is
            computed in line search.
        amax: upper bound of the value of the step size that is
            computed in line search.
        maxiter_linesearch: the maximum number of the iteration
            used in line search.
        rho_const: when computing :math:`1/x`, where :math:`x`
            is a scaler, sometimes it returns zero division error.
            In that case :math:`1/x` is replaced by ``rho_cost``.
        m: In parameters update of each step, it uses the info of
            the last ``m`` steps.
        gtol: If not None, it is used for determining if the
            opotimization has terminated successfully.
            If ``gtol`` is less than the infinity norm of the
            gradient of the cost function, the
            optimization is regarded to have terminated successfully.
            The infinity norm of the gradient :math:`g` is defined as
            :math:`||g||_{\infty} = \max\{|g_1|, |g_2|, \ldots, |g_n|\}`

    Refs:
        [1]: https://github.com/scipy/scipy/blob/master/scipy/optimize/optimize.py
        [2]: Jorge Nocedal and Stephen J. Wright.
            Numerical Optimization (Springer, New York, 2006).
    """

    def __init__(
        self,
        c1: float = 1e-4,
        c2: float = 0.4,
        amin: float = 1e-100,
        amax: float = 1e100,
        maxiter_linesearch: int = 20,
        rho_const: float = 1000.0,
        m: int = 5,
        gtol: Optional[float] = 1e-6,
    ):
        if not (0.0 < c1 < c2 < 1.0):
            raise ValueError(
                f"c1 and c2 must satisfy '0 < c1 < c2 < 1',"
                f" But got c1={c1:.5f} and c2={c2:.5f}"
            )
        if not rho_const > 0.0:
            raise ValueError(
                "Expected value for 'rho_const' has to be positive."
                f"But got {rho_const:.2f}"
            )
        self._c1 = c1
        self._c2 = c2
        self._amin = amin
        self._amax = amax
        self._maxiter_linesearch = maxiter_linesearch
        self._rho_const = rho_const
        self._m = m

        self._gtol: Optional[Callable[["npt.NDArray[np.float_]"], bool]] = None
        if gtol is not None:
            if not 0.0 < gtol:
                raise ValueError("ftol must be a positive float.")
            self._gtol = create_gtol(gtol)

    def get_init_state(self, init_params: Params) -> OptimizerStateLBFGS:
        params = readonly_array(np.array(init_params, dtype=float))
        zeros2d = readonly_array(np.zeros((self._m, len(params)), dtype=float))
        zeros = readonly_array(np.zeros(self._m, dtype=float))
        return OptimizerStateLBFGS(
            params=params,
            grad=zeros,
            p=zeros,
            s=zeros2d,
            y=zeros2d,
            rho=zeros,
            cost_prev=0.0,
            ind=0,
            a=zeros,
        )

    def step(
        self,
        state: OptimizerState,
        cost_function: CostFunction,
        grad_function: Optional[GradientFunction] = None,
    ) -> OptimizerStateLBFGS:
        if not isinstance(state, OptimizerStateLBFGS):
            raise ValueError("state must have type OptimizerStateLBFGS.")

        if grad_function is None:
            raise ValueError("LBFGS requires a gradient function.")

        funcalls = state.funcalls
        gradcalls = state.gradcalls
        niter = state.niter + 1

        params = state.params.copy()
        cost = state.cost
        cost_prev = state.cost_prev
        grad = state.grad.copy()
        p = state.p.copy()

        if niter == 1:
            cost = cost_function(params)
            funcalls += 1

            grad = grad_function(params)
            gradcalls += 1
            cost_prev = cost + cast(float, np.linalg.norm(grad)) / 2.0

            p = -grad

        funcalls, gradcalls, alpha, cost, cost_prev, grad_next = self._linesearch12(
            funcalls=funcalls,
            gradcalls=gradcalls,
            cost_function=cost_function,
            grad_function=grad_function,
            params=params,
            p=p,
            grad=grad,
            cost=cost,
            cost_prev=cost_prev,
        )

        if alpha is None:
            return replace(
                state,
                cost=cost,
                cost_prev=cost_prev,
                status=OptimizerStatus.FAILED,
                niter=niter,
                funcalls=funcalls,
                gradcalls=gradcalls,
                grad=readonly_array(grad),
                p=readonly_array(p),
            )

        params += alpha * p

        if grad_next is None:
            grad_next = grad_function(params)
            gradcalls += 1

        if self._gtol is not None and self._gtol(grad_next):
            return replace(
                state,
                params=readonly_array(params),
                cost=cost,
                cost_prev=cost_prev,
                status=OptimizerStatus.CONVERGED,
                niter=niter,
                funcalls=funcalls,
                gradcalls=gradcalls,
                grad=readonly_array(grad_next),
                p=readonly_array(p),
            )

        s = state.s.copy()
        y = state.y.copy()
        rho = state.rho.copy()
        ind = state.ind
        a = state.a.copy()

        s[ind] = alpha * p
        y[ind] = grad_next - grad

        rho_inv = np.dot(s[ind], y[ind])
        rho[ind] = self._rho_const if rho_inv == 0.0 else 1.0 / rho_inv

        grad = grad_next

        q = grad.copy()
        for i in reversed(range(ind + 1)):
            a[i] = np.dot(s[i], q) * rho[i]
            q -= a[i] * y[i]
        for i in reversed(range(ind + 1, self._m)):
            a[i] = np.dot(s[i], q) * rho[i]
            q -= a[i] * y[i]

        r = q / rho[ind] / np.linalg.norm(y[ind]) ** 2
        for i in range(ind + 1, self._m):
            b = rho[i] * np.dot(y[i], r)
            r += s[i] * (a[i] - b)
        for i in range(ind + 1):
            b = rho[i] * np.dot(y[i], r)
            r += s[i] * (a[i] - b)
        p = -r

        ind = (ind + 1) % self._m

        return OptimizerStateLBFGS(
            params=readonly_array(params),
            cost=cost,
            cost_prev=cost_prev,
            status=OptimizerStatus.SUCCESS,
            niter=niter,
            funcalls=funcalls,
            gradcalls=gradcalls,
            grad=readonly_array(grad),
            p=readonly_array(p),
            s=readonly_array(s),
            y=readonly_array(y),
            rho=readonly_array(rho),
            ind=ind,
            a=readonly_array(a),
        )

    def _linesearch12(
        self,
        funcalls: int,
        gradcalls: int,
        cost_function: CostFunction,
        grad_function: GradientFunction,
        params: Params,
        p: Params,
        grad: Params,
        cost: float,
        cost_prev: float,
        extra_condition: Optional[
            Callable[
                [float, "npt.NDArray[np.float_]", float, "npt.NDArray[np.float_]"], bool
            ]
        ] = None,
    ) -> tuple[int, int, float, float, float, "npt.NDArray[np.float_]"]:
        alpha, fc, gc, fval, old_fval, gval = line_search_wolfe1(
            f=cost_function,
            fprime=grad_function,
            xk=params,
            pk=p,
            gfk=grad,
            old_fval=cost,
            old_old_fval=cost_prev,
            c1=self._c1,
            c2=self._c2,
            amin=self._amin,
            amax=self._amax,
        )

        funcalls += fc
        gradcalls += gc

        if alpha is not None and extra_condition is not None:
            params_next = params + alpha * p
            if not extra_condition(alpha, params_next, fval, gval):
                alpha = None

        if alpha is None:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", LineSearchWarning)
                alpha, fc, gc, fval, old_fval, gval = line_search_wolfe2(
                    f=cost_function,
                    myfprime=grad_function,
                    xk=params,
                    pk=p,
                    gfk=grad,
                    old_fval=cost,
                    old_old_fval=cost_prev,
                    c1=self._c1,
                    c2=self._c2,
                    amax=self._amax,
                    extra_condition=extra_condition,
                    maxiter=self._maxiter_linesearch,
                )
            funcalls += fc
            gradcalls += gc

        return funcalls, gradcalls, alpha, fval, old_fval, gval
