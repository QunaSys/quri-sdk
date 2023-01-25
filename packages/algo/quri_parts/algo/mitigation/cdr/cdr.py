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
from quri_parts.circuit import (
    NonParametricQuantumCircuit,
    QuantumCircuit,
    QuantumGate,
    is_clifford,
)
from quri_parts.circuit.transpile import CliffordApproximationTranspiler
from quri_parts.core.estimator import (
    ConcurrentQuantumEstimator,
    Estimatable,
    Estimate,
    QuantumEstimator,
)
from quri_parts.core.operator import Operator
from quri_parts.core.state import GeneralCircuitQuantumState


class _Estimate(NamedTuple):
    value: complex
    error: float = np.nan  # The error of cdr has been not implemented yet.


#: Interface representing regression ansatz
RegressionMethod: TypeAlias = Callable[[float, Iterable[float], Iterable[float]], float]


def make_training_circuits(
    circuit: NonParametricQuantumCircuit,
    num_non_clifford_untouched: int,
    num_training_circuits: int = 10,
    seed: Optional[int] = None,
) -> list[NonParametricQuantumCircuit]:
    """Returns a list of near Clifford circuits obtained by replacing some non-
    Clifford gates in the input circuit by the nearest Clifford gates. The gate
    to be replaced is chosen at random.

    Args:
        circuit: Original circuit from which the near Clifford circuit is constructed.
        num_non_clifford_untouched: A number of non-Clifford gates that remain untouched
            in the replacing.
        num_training_circuits: Number of circuits to be returned.
        seed: Seed to choose which gates to replace with Clifford ones.
    """

    indices_non_clifford = [
        index for index, gate in enumerate(circuit.gates) if not is_clifford(gate)
    ]

    if len(indices_non_clifford) == 0:
        raise ValueError("No non-clifford gate in the input circuit.")

    num_to_replace = len(indices_non_clifford) - num_non_clifford_untouched

    if num_to_replace < 0:
        raise ValueError(
            "num_non_clifford_untouched must be less than the number of the\
            non-clifford gate in the circuit."
        )

    training_circuits: list[NonParametricQuantumCircuit] = []
    ram_seed = np.random.RandomState(seed)
    seed_list = list(ram_seed.randint(int(10e4), size=num_training_circuits))

    for i in range(num_training_circuits):
        ram = np.random.RandomState(seed_list[i])
        indices_rep = ram.choice(
            indices_non_clifford, size=num_to_replace, replace=False
        )

        new_gates: list[QuantumGate] = []
        for index, gate in enumerate(circuit.gates):
            if index in indices_rep:
                transpiler = CliffordApproximationTranspiler()
                transpiled_gates = transpiler.decompose(gate)
                new_gates += transpiled_gates
            else:
                new_gates.append(gate)
        training_circuits.append(QuantumCircuit(circuit.qubit_count, new_gates))

    return training_circuits


def create_polynomial_regression(order: int) -> RegressionMethod:
    """Returns a :class:`RegressionMethod` that gives a value which is
    evaluated by using the polynomial regression.

    Args:
        order: Order of the polynomial used for regression.
    """

    def polynomial_regression(
        values_noise: float,
        values_noise_training: Iterable[float],
        values_exact_training: Iterable[float],
    ) -> float:
        opt_result = polynomial_fitting(
            values_noise_training, values_exact_training, order, values_noise
        )
        return opt_result.value

    return polynomial_regression


def create_exp_regression(order: int) -> RegressionMethod:
    """Returns a :class:`RegressionMethod` that gives a value which
    is evaluated by using the curve regression and exponential function
    f(x) = a + b exp(p(x)), where p(x) is a polynomial of a given order.

    Args:
        order: Order of the polynomial used for regression.
    """

    def exp_regression(
        values_noise: float,
        values_noise_training: Iterable[float],
        values_exact_training: Iterable[float],
    ) -> float:
        opt_result = exp_fitting(
            values_noise_training, values_exact_training, order, values_noise
        )
        return opt_result.value

    return exp_regression


def create_exp_regression_with_const(order: int, constant: float) -> RegressionMethod:
    """Returns a :class:`RegressionMethod` that gives a value which
    is evaluated by using the curve regression and exponential function
    f(x) = constant + b exp(p(x)), where p(x) is a polynomial
    of a given order and constant is a known parameter (obtained as the
    infinite limit f(x->inf) when f(x) converges to a finite asymptotic value).

    Args:
        order: Order of the polynomial used for regression.
        constant: A constant deduced from asymptotic behavior f(x->inf).
    """

    def exp_regression_with_const(
        values_noise: float,
        values_noise_training: Iterable[float],
        values_exact_training: Iterable[float],
    ) -> float:
        opt_result = exp_fitting_with_const(
            values_noise_training, values_exact_training, order, constant, values_noise
        )
        return opt_result.value

    return exp_regression_with_const


def create_exp_regression_with_const_log(
    order: int, constant: float
) -> RegressionMethod:
    """Returns a :class:`RegressionMethod` that gives a value which
    is evaluated by using the log regression and exponential function
    f(x) = constant + b exp(p(x)), where p(x) is a polynomial
    of a given order and constant is a known parameter (obtained as the
    infinite limit f(x->inf) when f(x) converges to a finite asymptotic value).

    Args:
        order: Order of the polynomial used for regression.
        constant: A constant deduced from asymptotic behavior f(x->inf).
    """

    def exp_regression_with_const_log(
        values_noise: float,
        values_noise_training: Iterable[float],
        values_exact_training: Iterable[float],
    ) -> float:
        opt_result = exp_fitting_with_const_log(
            values_noise_training, values_exact_training, order, constant, values_noise
        )
        return opt_result.value

    return exp_regression_with_const_log


def cdr(
    obs: Estimatable,
    circuit: NonParametricQuantumCircuit,
    noisy_estimator: ConcurrentQuantumEstimator[GeneralCircuitQuantumState],
    exact_estimator: ConcurrentQuantumEstimator[GeneralCircuitQuantumState],
    regression_method: RegressionMethod,
    num_training_circuits: int = 10,
    fraction_of_replacement: float = 0.1,
    seed: Optional[int] = None,
) -> float:
    """Returns an error-mitigated expectation value of an observable by using
    clifford-data-regression (CDR).

    Args:
        obs: Observable for which the expected value is calculated.
        circuit: Circuit that CDR is applied to.
        noisy_estimator: A noisy estimator that computes the expectation value from
            obs and circuit.
        exact_estimator: An exact (noiseless) estimator that computes the expectation
            value from obs and training circuits. This is assumed a simulator that
            can do a fast simulation of the (almost) Clifford circuit.
        regression_method: Method used for regression.
        num_training_circuits: A number of training circuits to be used for CDR.
        fraction_of_replacement: Fraction of the number of non-Clifford gates that are
            to be replaced by Clifford gates.
        seed: Seed to choose which gates to replace with Clifford ones.
    """

    if isinstance(obs, Operator):
        if obs != obs.hermitian_conjugated():
            raise NotImplementedError(
                "Only the case that obs is a Hermitian has been implemented."
            )
    if fraction_of_replacement < 0 or 1 <= fraction_of_replacement:
        raise ValueError(
            "fraction_of_replacement must be larger than 0 and less than 1."
        )

    num_non_clifford = len([gate for gate in circuit.gates if not is_clifford(gate)])

    if num_non_clifford == 0:
        clifford_circuit_state = GeneralCircuitQuantumState(
            circuit.qubit_count, circuit
        )
        return [
            exp.value.real for exp in exact_estimator([obs], [clifford_circuit_state])
        ][0]

    num_to_replace = max(int(round(num_non_clifford * fraction_of_replacement)), 1)
    num_clifford_untouched = num_non_clifford - num_to_replace
    training_circuits = make_training_circuits(
        circuit, num_clifford_untouched, num_training_circuits, seed
    )

    circuit_state = GeneralCircuitQuantumState(circuit.qubit_count, circuit)
    values_noise = [exp.value.real for exp in noisy_estimator([obs], [circuit_state])][
        0
    ]
    training_circuit_states = [
        GeneralCircuitQuantumState(i.qubit_count, i) for i in training_circuits
    ]
    values_exact_training = [
        exp.value.real for exp in exact_estimator([obs], training_circuit_states)
    ]
    values_noise_training = [
        exp.value.real for exp in noisy_estimator([obs], training_circuit_states)
    ]

    return regression_method(
        values_noise, values_noise_training, values_exact_training
    ).real


def create_cdr_estimator(
    noisy_estimator: ConcurrentQuantumEstimator[GeneralCircuitQuantumState],
    exact_estimator: ConcurrentQuantumEstimator[GeneralCircuitQuantumState],
    regression_method: RegressionMethod,
    num_training_circuits: int = 10,
    fraction_of_replacement: float = 0.1,
    seed: Optional[int] = None,
) -> QuantumEstimator[GeneralCircuitQuantumState]:
    """Wrap the given :class:`ConcurrentQuantumEstimator` to create a
    :class:`QuantumEstimator` where clifford-data-regression is automatically
    applied to the result.

    Args:
        noisy_estimator: A noisy estimator that computes the expectation value from
            obs and circuit. This is assumed a given ConcurrentQuantumEstimator which
            will be wrapped.
        exact_estimator: An exact (noiseless) estimator that computes the expectation
            value from obs and training circuits. This is assumed a simulator that
            can do a fast simulation of the (almost) Clifford circuit.
        regression_method: Method used for regression.
        num_training_circuits: A number of training circuits to be used for CDR.
        fraction_of_replacement: Fraction of the number of non-Clifford gates that are
            to be replaced by Clifford gates.
        seed: Seed to choose which gates to replace with Clifford ones.
    """

    def cdr_estimator(
        obs: Estimatable, state: GeneralCircuitQuantumState
    ) -> Estimate[complex]:
        circuit = state.circuit
        cdr_value = cdr(
            obs,
            circuit,
            noisy_estimator,
            exact_estimator,
            regression_method,
            num_training_circuits,
            fraction_of_replacement,
            seed,
        )
        return _Estimate(value=cdr_value)

    return cdr_estimator
