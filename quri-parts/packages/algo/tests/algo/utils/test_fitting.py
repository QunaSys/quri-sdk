# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np

from quri_parts.algo.utils import (
    exp_fitting,
    exp_fitting_with_const,
    exp_fitting_with_const_log,
    polynomial_fitting,
)


def test_polynomial_fitting() -> None:
    ans_value = 7.0262899
    x_data = [1, 2.6, 4.4, 7.8, 9.1]
    y_data: list[float] = [13.4, 145.44224, 745.67024, 5302.63584, 9217.32389]
    fitted_result = polynomial_fitting(x_data, y_data, 4, 0.7)
    assert np.isclose(fitted_result.value, ans_value)


def test_exp_fitting() -> None:
    ans_value = 0.67311
    x_data = [0.1, 0.6, 0.8, 1.1, 1.3, 1.4, 1.7]
    y_data: list[float] = [
        0.44622,
        0.61393,
        0.74553,
        1.07954,
        1.45978,
        1.72633,
        3.05448,
    ]
    fitted_result = exp_fitting(x_data, y_data, 2, 0.7)
    np.isclose(fitted_result.value, ans_value)


def test_exp_fitting_with_const() -> None:
    ans_value = 0.33847636
    const = 0.1
    x_data = [0.1, 0.6, 0.8, 1.1, 1.3, 1.4, 1.7]
    y_data: list[float] = [0.23111, 0.30389, 0.38737, 0.7196, 1.36809, 2.04608, 9.75538]
    fitted_result = exp_fitting_with_const(x_data, y_data, 3, const, 0.7)
    np.isclose(fitted_result.value, ans_value)


def test_exp_fittings_with_const_log() -> None:
    ans_value = 0.33847636
    const = 0.1
    x_data = [0.1, 0.6, 0.8, 1.1, 1.3, 1.4, 1.7]
    y_data: list[float] = [
        0.23111439,
        0.30388787,
        0.38736989,
        0.71959757,
        1.3680939,
        2.04607575,
        9.75537647,
    ]
    fitted_result = exp_fitting_with_const_log(x_data, y_data, 3, const, 0.7)
    np.isclose(fitted_result.value, ans_value)
