# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import NamedTuple

from quri_parts.core.operator import Operator, PauliLabel


class ExponentialSinglePauli(NamedTuple):
    r"""A class representing exponential function of a Pauli operator.

    This class represents an exponential function of Pauli operators
    :math:`\exp(aP)` where :math:`a` is a coefficient and :math:`P` is a
    Pauli operator. Note that this coefficient can also be a complex
    number.
    """

    pauli: PauliLabel
    coefficient: complex


def _trotter_suzuki_decomposition_lowest_order(
    op: Operator, param: complex
) -> list[ExponentialSinglePauli]:
    exp_list = []
    for k, v in op.items():
        exp_list.append(ExponentialSinglePauli(pauli=k, coefficient=param * v / 2))
    lowest_order_list = []
    for each_circuit in exp_list:
        lowest_order_list.append(each_circuit)
    for each_circuit_inv in exp_list[::-1]:
        lowest_order_list.append(each_circuit_inv)
    return lowest_order_list


def _p_k(k: int) -> float:
    returned = 1 / (4 - 4 ** (1 / (2 * k - 1)))
    if returned > 1:
        raise ValueError(f"Invalid p_k = {returned}.")
    if returned < 0.25:
        raise ValueError(f"Invalid p_k = {returned}.")
    return returned


def trotter_suzuki_decomposition(
    op: Operator, param: complex, order: int
) -> list[ExponentialSinglePauli]:
    r"""Trotter-Suzuki decomposition [1], a recursive formula of the
    approximation that decomposes an exponential function of the sum of Pauli
    operators into a product of the exponential function of Pauli operators.
    The explicit formula for the sum of Pauli operator :math:`A=\sum_i A_i` is
    given as follows:

    .. math::
        S_{2k}(x)&=[S_{2k-2}(p_kx)]^2S_{2k-2}((1-4p_k)x)][S_{2k-2}(p_kx)]^2,\\
        S_2(x)&=\prod_{j=1}^me^{A_j x /2}\prod_{j'=m}^1 e^{A_{j'} x /2},\\
        p_k&=(4-4^{1/(2k-1)})^{-1},

    where :math:`k` is an order of this decomposition and :math:`A_i` is the Pauli
    operator.

    Args:
        op: An operator on the exponential.
        param: The overall coefficient of the exponential. This can be not only a
            real but also a complex number.
        order: The order of the Trotter-Suzuki decomposition. An integer that
            satisfies >= 1.

    Returns:
        List of ExponentialSinglePauli.

    Ref:
        [1]: M. Suzuki, Fractal decomposition of exponential operators with
        applications to many-body theories and Monte Carlo simulation,
        Phys. Lett. 146 319-323, 1990
    """

    if order < 1:
        raise ValueError("Invalid order.")
    if op.n_terms == 1:
        p, c = next(iter(op.items()))
        return [ExponentialSinglePauli(p, c * param)]
    if order == 1:
        return _trotter_suzuki_decomposition_lowest_order(op, param)
    else:
        exps_1 = trotter_suzuki_decomposition(op, _p_k(order) * param, order - 1)
        exps_2 = [
            ExponentialSinglePauli(pn, cn * (1 / _p_k(order) - 4)) for pn, cn in exps_1
        ]
        exp_list = exps_1 * 2 + exps_2 + exps_1 * 2
        return exp_list
