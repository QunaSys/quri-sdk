# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Callable, Iterable, NamedTuple, Optional

import numpy as np
from typing_extensions import TypeAlias

from quri_parts.algo.utils import (
    exp_fitting,
    exp_fitting_with_const,
    exp_fitting_with_const_log,
    polynomial_fitting,
)
from quri_parts.circuit import NonParametricQuantumCircuit, QuantumCircuit, inverse_gate
from quri_parts.core.estimator import (
    ConcurrentQuantumEstimator,
    Estimatable,
    Estimate,
    QuantumEstimator,
)
from quri_parts.core.operator import Operator
from quri_parts.core.state import GeneralCircuitQuantumState


class _Estimate(NamedTuple):
    value: float
    error: float = np.nan  # The error of zne has been not implemented yet.


#: Interface representing folding methods
FoldingMethod: TypeAlias = Callable[[NonParametricQuantumCircuit, float], list[int]]


#: Interface representing zero noise extrapolation methods
ZeroExtrapolationMethod: TypeAlias = Callable[[Iterable[float], Iterable[float]], float]


#: Interface representing scaling methods of circuit
CircuitScaling: TypeAlias = Callable[
    [NonParametricQuantumCircuit, float, FoldingMethod], NonParametricQuantumCircuit
]


def _get_residual_n_gates(
    circuit: NonParametricQuantumCircuit, scale_factor: float
) -> int:
    """Returns the number of gates that need to be additionally folded after
    the gates of the entire circuit has been folded.

    Args:
        circuit: Circuit to be folded.
        scale_factor: : Factor to scale the circuit. A real number that satisfies >= 1.
    """

    if scale_factor < 1:
        raise ValueError("Scale_factor must be greater than or equal to 1.")
    gate_num = len(circuit.gates)
    int_scale_factor = int((scale_factor - 1) / 2)
    diff = scale_factor - (2 * int_scale_factor + 1)
    return int((diff * gate_num) / 2)


# implementations of FoldingMethod
def create_folding_left() -> FoldingMethod:
    """Returns a :class:`FoldingMethod` that gives a list of indices of gates
    to be folded.

    Folding starts from the left of the circuit.
    """

    def folding_left(
        circuit: NonParametricQuantumCircuit, scale_factor: float
    ) -> list[int]:
        add_gate_num = _get_residual_n_gates(circuit, scale_factor)
        return list(range(add_gate_num))

    return folding_left


def create_folding_right() -> FoldingMethod:
    """Returns a :class:`FoldingMethod` that gives a list of indices of gates
    to be folded.

    Folding starts from the right of the circuit.
    """

    def folding_right(
        circuit: NonParametricQuantumCircuit, scale_factor: float
    ) -> list[int]:
        add_gate_num = _get_residual_n_gates(circuit, scale_factor)
        n_gates = len(circuit.gates)
        return list(range(n_gates - add_gate_num, n_gates))

    return folding_right


def create_folding_random(seed: Optional[int] = None) -> FoldingMethod:
    """Returns a :class:`FoldingMethod` that gives a list of indices of gates
    to be folded.

    The gates to be folded are chosen at random.

    Args:
        seed: Seed for random number generator.
    """

    def folding_random(
        circuit: NonParametricQuantumCircuit, scale_factor: float
    ) -> list[int]:
        add_gate_num = _get_residual_n_gates(circuit, scale_factor)
        list_random = np.random.RandomState(seed)
        add_gate_list = list(
            list_random.choice(range(add_gate_num), size=(add_gate_num), replace=False)
        )
        return add_gate_list

    return folding_random


# implementations of CircuitScaling
def scaling_circuit_folding(
    circuit: NonParametricQuantumCircuit,
    scale_factor: float,
    folding_method: FoldingMethod,
) -> NonParametricQuantumCircuit:
    """Returns a scaled circuit. If the scale factor is odd, all gates are
    folded regardless of the folding method.

    Args:
        circuit: Circuit to be folded.
        scale_factor: Factor to scale the circuit. A real number that satisfies >= 1.
        folding_method: Folding method applied to the circuit as a
            :class:`FoldingMethod`.
    """

    scaled_circuit = QuantumCircuit(circuit.qubit_count)
    num_folding_allgates = int((scale_factor - 1) / 2)
    add_gate_list = folding_method(circuit, scale_factor)

    for i, gate in enumerate(circuit.gates):
        inv_gate = inverse_gate(gate)
        scaled_circuit.add_gate(gate)
        for j in range(num_folding_allgates):
            scaled_circuit.add_gate(inv_gate)
            scaled_circuit.add_gate(gate)
        if i in add_gate_list:
            scaled_circuit.add_gate(inv_gate)
            scaled_circuit.add_gate(gate)
    return scaled_circuit


# implementations of ZeroExtrapolationMethod
def create_polynomial_extrapolate(order: int) -> ZeroExtrapolationMethod:
    """Returns a :class:`ZeroExtrapolationMethod` that gives a zero-noise
    extrapolated value which is evaluated by using the polynomial fitting.

    Args:
        order: Order of the polynomial used for fitting.
    """

    def polynomial_extrapolate(
        scale_factors: Iterable[float], exp_values: Iterable[float]
    ) -> float:
        opt_result = polynomial_fitting(scale_factors, exp_values, order, 0)
        return opt_result.parameters[0]

    return polynomial_extrapolate


def create_exp_extrapolate(order: int) -> ZeroExtrapolationMethod:
    """Returns a :class:`ZeroExtrapolationMethod` that gives a zero-noise extrapolated
    value which is evaluated by using the curve fitting and exponential ansatz
    f(x) = a + b exp(p(x)), where p(x) is a polynomial of a given order.

    Args:
        order: Order of the polynomial on the exponential used for fitting.
    """

    def exp_extrapolate(
        scale_factors: Iterable[float],
        exp_values: Iterable[float],
    ) -> float:
        return exp_fitting(scale_factors, exp_values, order, 0).value

    return exp_extrapolate


def create_exp_extrapolate_with_const(
    order: int, constant: float
) -> ZeroExtrapolationMethod:
    """Returns a :class:`ZeroExtrapolationMethod` that gives a zero-noise
    extrapolated value which is evaluated by using the curve fitting and
    exponential ansatz f(x) = constant + b exp(p(x)), where p(x) is a polynomial
    of a given order and constant is a known parameter (obtained as the
    infinite-noise limit f(x->inf)).

    Args:
        order: Order of the polynomial on the exponential used for fitting.
        constant: An infinite-noise limit parameter f(x->inf).
    """

    def exp_extrapolate_with_const(
        scale_factors: Iterable[float],
        exp_values: Iterable[float],
    ) -> float:
        opt_result = exp_fitting_with_const(
            scale_factors, exp_values, order, constant, 0
        )
        return opt_result.value

    return exp_extrapolate_with_const


def create_exp_extrapolate_with_const_log(
    order: int, constant: float
) -> ZeroExtrapolationMethod:
    """Returns a :class:`ZeroExtrapolationMethod` that gives a zero-noise
    extrapolated value which is evaluated by using the log fitting and
    exponential ansatz f(x) = constant + b exp(p(x)), where p(x) is a polynomial
    of a given order and constant is a known parameter (obtained as the
    infinite-noise limit f(x->inf)).

    Args:
        order: Order of the polynomial on the exponential used for fitting.
        constant: An infinite-noise limit parameter f(x->inf).
    """

    def exp_extrapolate_with_const_log(
        scale_factors: Iterable[float],
        exp_values: Iterable[float],
    ) -> float:
        opt_result = exp_fitting_with_const_log(
            scale_factors, exp_values, order, constant, 0
        )
        return opt_result.value

    return exp_extrapolate_with_const_log


def zne(
    obs: Estimatable,
    circuit: NonParametricQuantumCircuit,
    estimator: ConcurrentQuantumEstimator[GeneralCircuitQuantumState],
    scale_factors: Iterable[float],
    extrapolate_method: ZeroExtrapolationMethod,
    folding_method: FoldingMethod,
) -> float:
    """Returns an error-mitigated expectation value of an observable by using
    zero-noise extrapolation (zne).

    Args:
        obs: Observable for which the expected value is calculated.
        circuit: Circuit that ZNE is applied to.
        estimator: An estimator that computes the expectation value from
            obs and circuit.
        scale_factors: Factor to scale the circuit. An real number that satisfies >= 1.
        extrapolate_method: :class:`ZeroExtrapolationMethod` that determine the
            method of extrapolation.
        folding_method: :class:`FoldingMethod` that determines the method of folding.
    """

    if isinstance(obs, Operator):
        if obs != obs.hermitian_conjugated():
            raise NotImplementedError(
                "Only the case that obs is a Hermitian has been implemented."
            )

    exp_values: Iterable[float] = []
    circuit_states = []
    for i in scale_factors:
        scaled_circuit = scaling_circuit_folding(circuit, i, folding_method)
        circuit_states.append(
            GeneralCircuitQuantumState(scaled_circuit.qubit_count, scaled_circuit)
        )
    exp_values = [exp.value.real for exp in estimator([obs], circuit_states)]

    return extrapolate_method(scale_factors, exp_values)


def richardson_extrapolation(
    obs: Estimatable,
    circuit: NonParametricQuantumCircuit,
    estimator: ConcurrentQuantumEstimator[GeneralCircuitQuantumState],
    scale_factors: Iterable[float],
    folding_method: FoldingMethod,
) -> float:
    """Returns an error-mitigated expectation value of an observable by using
    zne.

    The order of polynomial = len(list(scale_factors)) - 1.

    Args:
        obs: Observable for which the expected value is calculated.
        circuit: Circuit that apply zne.
        estimator: An estimator that computes the expectation value from
            obs and circuit.
        scale_factors: Factor to scale the circuit. A real number that satisfies >= 1.
        folding_method: :class:`FoldingMethod` that determines the method of folding.
    """

    richardson_extrapolate = create_polynomial_extrapolate(
        order=len(list(scale_factors)) - 1
    )
    return zne(
        obs, circuit, estimator, scale_factors, richardson_extrapolate, folding_method
    )


def create_zne_estimator(
    estimator: ConcurrentQuantumEstimator[GeneralCircuitQuantumState],
    scale_factors: Iterable[float],
    extrapolate_method: ZeroExtrapolationMethod,
    folding_method: FoldingMethod,
) -> QuantumEstimator[GeneralCircuitQuantumState]:
    """Wrap the given :class:`ConcurrentQuantumEstimator` to create a
    :class:`QuantumEstimator` where zero-noise mitigation is automatically
    applied to the result.

    Args:
        estimator: An estimator that computes the expectation value from
            obs and circuit.
        scale_factors: Factor to scale the circuit. A real number that satisfies >= 1.
        extrapolate_method: :class:`ZeroExtrapolationMethod` that determine the
            method of extrapolation.
        folding_method: :class:`FoldingMethod` that determines the method of folding.
    """

    def zne_estimator(
        obs: Estimatable, state: GeneralCircuitQuantumState
    ) -> Estimate[float]:
        circuit = state.circuit
        zne_value = zne(
            obs,
            circuit,
            estimator,
            scale_factors,
            extrapolate_method,
            folding_method,
        )
        return _Estimate(value=zne_value)

    return zne_estimator
