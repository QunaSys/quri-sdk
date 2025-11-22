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

from quri_parts.algo.optimizer import (
    AdaBelief,
    Adam,
    GradientFunction,
    OptimizerStatus,
    Params,
)
from quri_parts.algo.optimizer.adam import OptimizerStateAdam


def adam_update(
    lr: float,
    eps: float,
    betas: tuple[float, float],
    grad_fn: GradientFunction,
    params: Params,
    m: Params,
    v: Params,
    niter: int,
    adabelief: bool = False,
) -> tuple[Params, Params, Params]:
    t = niter + 1
    grad = grad_fn(params)
    beta1, beta2 = betas
    m_t = beta1 * m + (1 - beta1) * grad
    if adabelief:
        v_t = beta2 * v + (1 - beta2) * (grad - m_t) ** 2 + eps
    else:
        v_t = beta2 * v + (1 - beta2) * grad**2
    m_hat_t = m_t / (1 - beta1**t)
    v_hat_t = v_t / (1 - beta2**t)
    params_t = params - lr * m_hat_t / (np.sqrt(v_hat_t) + eps)
    return params_t, m_t, v_t


lr = 1e-3
betas = (0.8, 0.88)
eps = 1e-9

params = np.random.rand(5)
m = np.random.rand(len(params))
v = np.random.rand(len(params))


def cost_fn(params: Params) -> float:
    return cast(float, 0.1 * np.sum(params))


grad = np.random.rand(len(params))


def grad_fn(params: Params) -> Params:
    return grad


class TestAdam:
    def test_step(self) -> None:
        optimizer = Adam(lr, betas, eps, ftol=None)
        state = OptimizerStateAdam(
            params=params,
            niter=14,
            funcalls=20,
            gradcalls=17,
            m=m,
            v=v,
        )

        next_state = optimizer.step(state, cost_fn, grad_fn)

        next_params, next_m, next_v = adam_update(
            lr, eps, betas, grad_fn, params, m, v, state.niter
        )

        assert np.allclose(next_state.params, next_params)
        assert np.allclose(next_state.m, next_m)
        assert np.allclose(next_state.v, next_v)
        assert next_state.cost == pytest.approx(cost_fn(next_params))
        assert next_state.niter == state.niter + 1
        assert next_state.funcalls == state.funcalls + 1
        assert next_state.gradcalls == state.gradcalls + 1

    def test_initial(self) -> None:
        optimizer = Adam(lr, betas, eps, ftol=None)
        state = optimizer.get_init_state(params)
        next_state = optimizer.step(state, cost_fn, grad_fn)
        assert next_state.funcalls == 2
        assert next_state.gradcalls == 1

    def test_ftol(self) -> None:
        optimizer = Adam(lr, betas, eps, ftol=1e-1)
        state = optimizer.get_init_state(params)
        assert state.status == OptimizerStatus.SUCCESS

        i = 0

        def cost_fn(params: Params) -> float:
            nonlocal i
            i += 1
            if i == 1:
                return 0.5
            elif i == 2:
                return 0.4
            elif i == 3:
                return 0.37
            else:
                return 0.0

        state = optimizer.step(state, cost_fn, grad_fn)
        assert state.funcalls == 2
        assert state.status == OptimizerStatus.SUCCESS
        state = optimizer.step(state, cost_fn, grad_fn)
        assert state.funcalls == 3
        assert state.status == OptimizerStatus.CONVERGED


class TestAdaBelief:
    def test_step(self) -> None:
        optimizer = AdaBelief(lr, betas, eps, ftol=None)
        state = OptimizerStateAdam(
            params=params,
            niter=14,
            funcalls=20,
            gradcalls=17,
            m=m,
            v=v,
        )

        next_state = optimizer.step(state, cost_fn, grad_fn)

        next_params, next_m, next_v = adam_update(
            lr, eps, betas, grad_fn, params, m, v, state.niter, adabelief=True
        )

        assert np.allclose(next_state.params, next_params)
        assert np.allclose(next_state.m, next_m)
        assert np.allclose(next_state.v, next_v)
        assert next_state.cost == pytest.approx(cost_fn(next_params))
        assert next_state.niter == state.niter + 1
        assert next_state.funcalls == state.funcalls + 1
        assert next_state.gradcalls == state.gradcalls + 1
