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

from quri_parts.algo.optimizer import SPSA, OptimizerStatus, Params


def cost_fn(params: Params) -> float:
    return cast(float, sum(np.sin(params)))


init_params: Params = np.asarray(
    [0.72018845, 8.02917029, 0.01004912, 5.80557493, 2.99575596], dtype=float
)
a = 2 * np.pi / 10
c = 0.1
alpha = 0.6
gamma = 0.101
A = 10 * 10
ftol = 1e-5
rng_seed = 1019328


class TestSPSA:
    def test_step_success(self) -> None:
        optimizer = SPSA(
            a=a, c=c, alpha=alpha, gamma=gamma, A=A, ftol=ftol, rng_seed=rng_seed
        )
        state = optimizer.get_init_state(init_params)

        niter = 1
        for i in range(niter):
            state = optimizer.step(state, cost_fn)

        assert state.gradcalls == 0
        assert state.funcalls == 4

        assert state.status == OptimizerStatus.SUCCESS
        assert state.niter == niter

    def test_step_converge(self) -> None:
        optimizer = SPSA(
            a=a, c=c, alpha=alpha, gamma=gamma, A=A, ftol=ftol, rng_seed=rng_seed
        )
        state = optimizer.get_init_state(init_params)

        niter = 330
        for i in range(niter):
            state = optimizer.step(state, cost_fn)

        assert state.gradcalls == 0
        assert state.funcalls == 991

        assert state.status == OptimizerStatus.CONVERGED
        assert state.niter == niter
        assert state.cost == pytest.approx(-state.n_params, abs=1e-4)

    def test_initial(self) -> None:
        optimizer = SPSA(a=a, c=c, alpha=alpha, gamma=gamma, A=A, ftol=ftol)
        state = optimizer.get_init_state(init_params)
        next_state = optimizer.step(state, cost_fn)

        assert next_state.niter == 1
        assert next_state.gradcalls == 0
        assert next_state.funcalls == 4

    def test_init_value_error(self) -> None:
        with pytest.raises(ValueError):
            SPSA(a=-0.3)
        with pytest.raises(ValueError):
            SPSA(alpha=-0.1)
