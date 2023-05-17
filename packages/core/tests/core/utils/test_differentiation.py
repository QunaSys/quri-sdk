# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Sequence

import numpy as np

from quri_parts.core.operator import (
    PAULI_IDENTITY,
    Operator,
    is_ops_close,
    pauli_label,
    zero,
)
from quri_parts.core.utils.differentiation import (
    backward_difference_gradient_formula,
    backward_difference_hessian_formula,
    central_difference_gradient_formula,
    central_difference_hessian_formula,
    forward_difference_gradient_formula,
    forward_difference_hessian_formula,
    gradient,
    hessian,
    numerical_operator_gradient,
    numerical_operator_hessian,
)


def f(params: Sequence[float]) -> float:
    return (
        params[0] * params[0] * params[0] * 0.1
        - params[1] * params[1] * params[1] * 0.2
        + params[2] * params[2] * params[2] * 0.3
        + params[0] * params[0] * params[1] * 0.4
        - params[0] * params[1] * params[1] * 0.5
        + params[1] * params[1] * params[2] * 0.6
        - params[0] * params[2] * params[2] * 0.7
    )


def test_forward_difference_gradient_formula() -> None:
    params = [1, 2, 3]
    expected = [-6.4, 3.2, 6.3]
    assert np.allclose(forward_difference_gradient_formula(f, params), expected)


def test_backward_difference_gradient_formula() -> None:
    params = [1, 2, 3]
    expected = [-6.4, 3.2, 6.3]
    assert np.allclose(backward_difference_gradient_formula(f, params), expected)


def test_central_difference_gradient_formula() -> None:
    params = [1, 2, 3]
    expected = [-6.4, 3.2, 6.3]
    assert np.allclose(central_difference_gradient_formula(f, params), expected)


def test_forward_difference_hessian_formula() -> None:
    params = [1, 2, 3]
    expected = [[2.2, -1.2, -4.2], [-1.2, 0.2, 2.4], [-4.2, 2.4, 4.0]]
    assert np.allclose(
        forward_difference_hessian_formula(f, params), expected, rtol=1e-3
    )


def test_backward_difference_hessian_formula() -> None:
    params = [1, 2, 3]
    expected = [[2.2, -1.2, -4.2], [-1.2, 0.2, 2.4], [-4.2, 2.4, 4.0]]
    assert np.allclose(
        backward_difference_hessian_formula(f, params), expected, rtol=1e-3
    )


def test_central_difference_hessian_formula() -> None:
    params = [1, 2, 3]
    expected = [[2.2, -1.2, -4.2], [-1.2, 0.2, 2.4], [-4.2, 2.4, 4.0]]
    assert np.allclose(
        central_difference_hessian_formula(f, params), expected, rtol=1e-3
    )


def test_gradient() -> None:
    params = [1, 2, 3]
    step = 1e-3
    assert np.allclose(
        gradient(f, params, step),
        central_difference_gradient_formula(f, params, step=step),
    )


def test_hessian() -> None:
    params = [1, 2, 3]
    step = 1e-3
    assert np.allclose(
        hessian(f, params, step),
        central_difference_hessian_formula(f, params, step=step),
    )


def _h_generator(params: Sequence[float]) -> Operator:
    return Operator(
        {
            PAULI_IDENTITY: 1.0 * params[0],
            pauli_label("Z0"): 2.0 * params[0] * params[1],
            pauli_label("Z1"): 3.0 * params[2] ** 2,
            pauli_label("Z2"): 4.0 * params[3] ** 3,
            pauli_label("Z0 Z1"): 5.0 * (params[4] - 1.0),
            pauli_label("Z0 Z2"): 6.0 * (params[5] - params[4]),
        }
    )


def test_numerical_operator_gradient() -> None:
    params = [1, 2, 3, 4, 5, 6]
    expected = [
        Operator({PAULI_IDENTITY: 1.0, pauli_label("Z0"): 2 * params[1]}),
        Operator({pauli_label("Z0"): 2.0 * params[0]}),
        Operator({pauli_label("Z1"): 6.0 * params[2]}),
        Operator({pauli_label("Z2"): 12.0 * params[3] ** 2}),
        Operator({pauli_label("Z0 Z1"): 5.0, pauli_label("Z0 Z2"): -6.0}),
        Operator({pauli_label("Z0 Z2"): 6.0}),
    ]
    assert np.all(
        [
            is_ops_close(res, exp)
            for res, exp in zip(
                numerical_operator_gradient(params, _h_generator), expected
            )
        ]
    )

    params = [0, 0, 0, 0, 0, 0]
    expected = [
        zero(),
        zero(),
        zero(),
        zero(),
        Operator({pauli_label("Z0 Z1"): 5.0, pauli_label("Z0 Z2"): -6.0}),
        Operator({pauli_label("Z0 Z2"): 6.0}),
    ]
    assert np.all(
        [
            is_ops_close(res, exp)
            for res, exp in zip(
                numerical_operator_gradient(params, _h_generator, atol=1.01), expected
            )
        ]
    )


def test_numerical_operator_hessian() -> None:
    params = [1, 2, 3, 4, 5, 6]
    expected = [
        [
            zero(),
            Operator({pauli_label("Z0"): 2.0}),
            zero(),
            zero(),
            zero(),
            zero(),
        ],
        [
            Operator({pauli_label("Z0"): 2.0}),
            zero(),
            zero(),
            zero(),
            zero(),
            zero(),
        ],
        [
            zero(),
            zero(),
            Operator({pauli_label("Z1"): 6.0}),
            zero(),
            zero(),
            zero(),
        ],
        [zero(), zero(), zero(), Operator({pauli_label("Z2"): 96.0}), zero(), zero()],
        [zero(), zero(), zero(), zero(), zero(), zero()],
        [zero(), zero(), zero(), zero(), zero(), zero()],
    ]
    res = numerical_operator_hessian(params, _h_generator, step=1e-3)
    assert np.all(
        [
            [
                is_ops_close(res[i][j], expected[i][j], atol=1e-5)
                for j in range(len(params))
            ]
            for i in range(len(params))
        ]
    )

    params = [0, 0, 0, 0, 0, 0]
    expected = [
        [
            zero(),
            Operator({pauli_label("Z0"): 2.0}),
            zero(),
            zero(),
            zero(),
            zero(),
        ],
        [
            Operator({pauli_label("Z0"): 2.0}),
            zero(),
            zero(),
            zero(),
            zero(),
            zero(),
        ],
        [
            zero(),
            zero(),
            Operator({pauli_label("Z1"): 6.0}),
            zero(),
            zero(),
            zero(),
        ],
        [zero(), zero(), zero(), zero(), zero(), zero()],
        [zero(), zero(), zero(), zero(), zero(), zero()],
        [zero(), zero(), zero(), zero(), zero(), zero()],
    ]
    res = numerical_operator_hessian(params, _h_generator, step=1e-3)
    print(res)
    assert np.all(
        [
            [
                is_ops_close(res[i][j], expected[i][j], atol=1e-5)
                for j in range(len(params))
            ]
            for i in range(len(params))
        ]
    )
