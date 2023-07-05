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
from typing import TYPE_CHECKING, Callable, Optional, cast

import numpy as np
from scipy.optimize import curve_fit

if TYPE_CHECKING:
    import numpy.typing as npt  # noqa: F401

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
class OptimizerStateNFT(OptimizerState):
    pass


class NFTBase(Optimizer):
    def __init__(
        self,
        randomize: bool = False,
        ftol: Optional[float] = 1e-5,
    ):
        self._randomize = randomize
        self._ftol: Optional[Callable[[float, float], bool]] = None
        if ftol is not None:
            if not 0.0 < ftol:
                raise ValueError("ftol must be a positive float.")
            self._ftol = create_ftol(ftol)

    def get_init_state(self, init_params: Params) -> OptimizerStateNFT:
        params = readonly_array(np.array(init_params, dtype=float))
        return OptimizerStateNFT(
            params=params,
        )

    def step(
        self,
        state: OptimizerState,
        cost_function: CostFunction,
        grad_function: Optional[GradientFunction] = None,
    ) -> OptimizerStateNFT:
        if not isinstance(state, OptimizerStateNFT):
            raise ValueError('state must have type "OptimizerStateNFT".')

        funcalls = state.funcalls
        niter = state.niter + 1

        if niter == 1:
            cost_prev = cost_function(state.params)
            funcalls += 1
        else:
            cost_prev = state.cost

        if self._randomize:
            index_list = np.random.permutation(np.arange(state.n_params))
        else:
            index_list = np.arange(state.n_params)

        params, cost, funcalls = self._param_step(
            state=OptimizerState(
                params=state.params,
                cost=cost_prev,
                status=state.status,
                niter=state.niter,
                funcalls=funcalls,
                gradcalls=state.gradcalls,
            ),
            index_list=index_list,
            cost_function=cost_function,
        )

        if self._ftol and self._ftol(cost, cost_prev):
            status = OptimizerStatus.CONVERGED
        else:
            status = OptimizerStatus.SUCCESS

        return OptimizerStateNFT(
            params=readonly_array(params),
            cost=cost,
            status=status,
            niter=niter,
            funcalls=funcalls,
            gradcalls=state.gradcalls,
        )

    @abstractmethod
    def _param_step(
        self,
        state: OptimizerState,
        index_list: "npt.NDArray[np.int64]",
        cost_function: CostFunction,
    ) -> tuple[Params, float, int]:
        pass


class NFT(NFTBase):
    """Nakanishi-Fujii-Todo optimization algorithm proposed in [1]. The
    algorithms optimizes the cost function sequentially with respect to the
    parameters.

    Args:
        randomize: If ``True``, the order of the parameters over which
            the cost function is optimized will be random. If ``False``,
            the order will be the same as the order of the array of initial
            parameters.
        reset_interval: minimum value of the cost function is evaluated by
            direct function-call only once in ``reset_interval`` times.
        eps: a small number used for avoiding zero division.
        ftol: If not None, judge convergence by cost function tolerance.
            See :func:`~.tolerance.ftol` for details.

    Ref:
        [1]: Ken M. Nakanishi, Keisuke Fujii, and Synge Todo,
            Sequential minimal optimization for quantum-classical hybrid algorithms,
            Phys. Rev. Research 2, 043158 (2020).
    """

    def __init__(
        self,
        randomize: bool = False,
        reset_interval: Optional[int] = 32,
        eps: float = 1e-32,
        ftol: Optional[float] = 1e-5,
    ):
        super().__init__(randomize, ftol)
        self._reset_interval = reset_interval

        if not eps >= 0.0:
            raise ValueError("'eps' must be zero or larger than zero.")
        self._eps = eps

    def _param_step(
        self,
        state: OptimizerState,
        index_list: "npt.NDArray[np.int64]",
        cost_function: CostFunction,
    ) -> tuple[Params, float, int]:
        params = state.params.copy()
        funcalls = state.funcalls
        cost = state.cost

        for i, idx in enumerate(index_list):
            if i == 0 and state.niter == 0:
                z0 = cost
            elif (
                self._reset_interval is not None
                and (i + state.niter * state.n_params) % self._reset_interval == 0
            ):
                z0 = cost_function(params)
                funcalls += 1
            else:
                z0 = cost

            p = params.copy()
            p[idx] = params[idx] + np.pi / 2.0
            z1 = cost_function(p)
            funcalls += 1

            p = params.copy()
            p[idx] = params[idx] - np.pi / 2.0
            z3 = cost_function(p)
            funcalls += 1

            z2 = z1 + z3 - z0
            c = (z1 + z3) / 2.0
            a = np.sqrt((z0 - z2) ** 2 + (z1 - z3) ** 2) / 2.0
            b = (
                np.arctan((z1 - z3) / ((z0 - z2) + self._eps * (z0 == z2)))
                + params[idx]
            )
            b += 0.5 * np.pi + 0.5 * np.pi * np.sign((z0 - z2) + self._eps * (z0 == z2))

            params[idx] = b
            cost = c - a  # value after minimization

        return params, cost, funcalls


class NFTfit(NFTBase):
    """Basically the same as Nakanishi-Fujii-Todo optimization algorithm [1].
    The difference from ``NFT`` class is that ``NFTfit`` uses SciPy fitting
    function ``curve_fit`` for parameters update.

    Args:
        randomize: If ``True``, the order of the parameters over which
            the cost function is optimized will be random. If ``False``,
            the order will be the same as the order of the array of initial
            parameters.
        n_point: Number of values of cost function for function fitting
            using SciPy fitting function ``curve_fit``.
        ftol: If not None, judge convergence by cost function tolerace.
            See :func:`~.tolerance.ftol` for details.

    Ref:
        [1]: Ken M. Nakanishi, Keisuke Fujii, and Synge Todo,
            Sequential minimal optimization for quantum-classical hybrid algorithms,
            Phys. Rev. Research 2, 043158 (2020).
    """

    def __init__(
        self,
        randomize: bool = False,
        n_points: int = 3,
        ftol: Optional[float] = 1e-5,
    ):
        super().__init__(randomize, ftol)

        self._n_points = n_points

    def _param_step(
        self,
        state: OptimizerState,
        index_list: "npt.NDArray[np.int64]",
        cost_function: CostFunction,
    ) -> tuple[Params, float, int]:
        x_data = np.pi * np.linspace(-1.0, 1.0, num=self._n_points, endpoint=False)
        y_data: "npt.NDArray[np.float_]" = np.zeros(self._n_points, dtype=float)

        params = state.params.copy()
        funcalls = state.funcalls

        for idx in index_list:
            for i in range(self._n_points):
                params[idx] = x_data[i]
                y_data[i] = cost_function(params)
                funcalls += 1
            # fit function: `A * cos(x-B) + C`
            # If `A > 0`, it is minimized at `x = B+pi`,
            # while if `A < 0` minimum at `x=b`
            fit_result, _ = curve_fit(_fit_func, x_data, y_data)
            a, b, _ = fit_result
            params[idx] = b + np.pi if a > 0.0 else b

        cost = cost_function(params)
        funcalls += 1

        return params, cost, funcalls


def _fit_func(x: float, a: float, b: float, c: float) -> float:
    """Fit with function in the form of.

    `A * cos(x-B) + C`
    If `A > 0` it is minimized at `x = B+pi`,
    while if `A < 0` minimum at `x=b`
    """
    return a * cast(float, np.cos(x - b)) + c
