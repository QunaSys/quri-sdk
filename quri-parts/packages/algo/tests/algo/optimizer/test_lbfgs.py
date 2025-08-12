# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import cast

import numpy as np
import pytest

from quri_parts.algo.optimizer import LBFGS, OptimizerStatus, Params


def cost_fn(params: Params) -> float:
    return cast(float, np.sum(np.sin(params)))


def grad_fn(params: Params) -> Params:
    return cast(Params, np.cos(params))


init_params: Params = np.asarray(
    [
        2.98593534,
        3.99004521,
        15.78411547,
        19.12760393,
        17.16611739,
        12.66622967,
        2.1830969,
        16.3386115,
        11.83433777,
        18.68013378,
        16.41234575,
        9.91515058,
        17.2384007,
        14.5397728,
        10.99869229,
        9.36807355,
        16.72513478,
        10.59205312,
        7.76694119,
        13.0476064,
    ],
    dtype=float,
)


class TestLBFGS:
    def test_step_success(self) -> None:
        optimizer = LBFGS(gtol=None)
        state = optimizer.get_init_state(init_params)
        assert state.status == OptimizerStatus.SUCCESS

        niter = 1
        for i in range(niter):
            state = optimizer.step(state, cost_fn, grad_fn)

        assert state.status == OptimizerStatus.SUCCESS
        assert state.niter == niter
        assert state.funcalls == 3
        assert state.gradcalls == 3

    def test_step_converge(self) -> None:
        gtol = 1e-5
        optimizer = LBFGS(gtol=gtol)
        state = optimizer.get_init_state(init_params)
        assert state.status == OptimizerStatus.SUCCESS

        niter = 10
        for i in range(niter):
            state = optimizer.step(state, cost_fn, grad_fn)

        assert state.status == OptimizerStatus.CONVERGED
        assert state.cost == pytest.approx(-state.n_params)
        assert np.amax(np.abs(state.grad)) <= gtol
        assert state.niter == niter
        assert state.funcalls == 13
        assert state.gradcalls == 13

    def test_initial(self) -> None:
        optimizer = LBFGS(gtol=1e-10)
        state = optimizer.get_init_state(init_params)
        next_state = optimizer.step(state, cost_fn, grad_fn)

        assert next_state.niter == 1
        assert next_state.funcalls == 3
        assert next_state.gradcalls == 3

    def test_init_value_error(self) -> None:
        with pytest.raises(ValueError):
            LBFGS(c1=1e-2, c2=1e-3)
        with pytest.raises(ValueError):
            LBFGS(rho_const=-10.0)
        with pytest.raises(ValueError):
            LBFGS(gtol=-1e-4)
