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

from quri_parts.algo.optimizer import NFT, NFTfit, OptimizerStatus, Params


def cost_fn(params: Params) -> float:
    return cast(float, sum(np.sin(params)))


init_params: Params = np.asarray(
    [5.672343, 0.3599771, 4.3338387, 0.3365039, 7.2604063], dtype=float
)


class TestNFT:
    def test_step_success(self) -> None:
        optimizer = NFT(randomize=True, reset_interval=3, ftol=1e-4)
        state = optimizer.get_init_state(init_params)

        niter = 1
        for i in range(niter):
            state = optimizer.step(state, cost_fn)

        assert state.gradcalls == 0
        assert state.funcalls == 12  # 1 init + (5 params * 2) * 1 iter + 1 reset

        assert state.status == OptimizerStatus.SUCCESS
        assert state.niter == niter

    def test_step_converge(self) -> None:
        optimizer = NFT(randomize=False, reset_interval=7, ftol=1e-4)
        state = optimizer.get_init_state(init_params)

        niter = 2
        for i in range(niter):
            state = optimizer.step(state, cost_fn)

        assert state.gradcalls == 0
        assert state.funcalls == 22  # 1 init + (5 params * 2) * 2 iter + 1 reset

        assert state.status == OptimizerStatus.CONVERGED
        assert state.niter == niter
        assert state.cost == pytest.approx(-state.n_params, abs=1e-4)

    def test_initial(self) -> None:
        optimizer = NFT(randomize=False, reset_interval=None, ftol=None)
        state = optimizer.get_init_state(init_params)
        next_state = optimizer.step(state, cost_fn)

        assert next_state.niter == 1
        assert next_state.gradcalls == 0
        assert next_state.funcalls == 11  # 1 init + (5 params * 2) * 1 iter

    def test_init_value_error(self) -> None:
        with pytest.raises(ValueError):
            NFT(ftol=-1e-4)
        with pytest.raises(ValueError):
            NFT(eps=-1e-30)


class TestNFTfit:
    def test_step_success(self) -> None:
        optimizer = NFTfit(randomize=True, n_points=5, ftol=1e-5)
        state = optimizer.get_init_state(init_params)
        assert state.status == OptimizerStatus.SUCCESS

        niter = 1
        for i in range(niter):
            state = optimizer.step(state, cost_fn)

        assert state.gradcalls == 0
        assert state.funcalls == 27  # 1 init + (5 params * 5 points + 1) * 1 iter

        assert state.status == OptimizerStatus.SUCCESS
        assert state.niter == niter

    def test_step_converge(self) -> None:
        optimizer = NFTfit(randomize=False, n_points=5, ftol=1e-5)
        state = optimizer.get_init_state(init_params)
        assert state.status == OptimizerStatus.SUCCESS

        niter = 2
        for i in range(niter):
            state = optimizer.step(state, cost_fn)

        assert state.gradcalls == 0
        assert state.funcalls == 53  # 1 init + (5 params * 5 points + 1) * 2 iter

        assert state.status == OptimizerStatus.CONVERGED
        assert state.niter == niter
        assert state.cost == pytest.approx(-state.n_params, abs=1e-4)

    def test_initial(self) -> None:
        optimizer = NFTfit(randomize=False, n_points=7, ftol=None)
        state = optimizer.get_init_state(init_params)
        next_state = optimizer.step(state, cost_fn)

        assert next_state.niter == 1
        assert next_state.gradcalls == 0
        assert next_state.funcalls == 37

    def test_init_value_error(self) -> None:
        with pytest.raises(ValueError):
            NFTfit(ftol=-1e-4)
