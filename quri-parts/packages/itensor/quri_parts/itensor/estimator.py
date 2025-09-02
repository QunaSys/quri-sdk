# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections.abc import Collection, Iterable, Sequence
from typing import TYPE_CHECKING, Any, Callable, NamedTuple, Optional

import juliacall
import numpy as np
from juliacall import Main as jl
from typing_extensions import TypeAlias

from quri_parts.core.estimator import (
    ConcurrentParametricQuantumEstimator,
    ConcurrentQuantumEstimator,
    Estimatable,
    Estimate,
    ParametricQuantumEstimator,
    QuantumEstimator,
    create_parametric_estimator,
)
from quri_parts.core.operator import zero
from quri_parts.core.state import CircuitQuantumState, ParametricCircuitQuantumState
from quri_parts.core.utils.concurrent import execute_concurrently
from quri_parts.itensor.load_itensor import ensure_itensor_loaded

from .circuit import convert_circuit
from .operator import convert_operator

if TYPE_CHECKING:
    from concurrent.futures import Executor


class _Estimate(NamedTuple):
    value: complex
    error: float = np.nan


#: A type alias for state classes supported by ITensor estimators.
#: ITensor estimators support circuit states.
ITensorStateT: TypeAlias = CircuitQuantumState

#: A type alias for parametric state classes supported by ITensor estimators.
#: ITensor estimators support circuit states.
ITensorParametricStateT: TypeAlias = ParametricCircuitQuantumState


def _estimate(
    operator: Estimatable, state: ITensorStateT, **kwargs: Any
) -> Estimate[complex]:
    if operator == zero():
        return _Estimate(value=0.0, error=0.0)
    qubits = state.qubit_count
    s: juliacall.VectorValue = jl.siteinds("Qubit", qubits)
    psi: juliacall.AnyValue = jl.init_state(s, qubits)

    # create ITensor circuit
    circuit = convert_circuit(state.circuit, s)

    # create ITensor operator
    op = convert_operator(operator, s)

    # apply circuit
    psi = jl.apply(circuit, psi, **kwargs)

    # calculate expectation value
    error = 0.0
    if any(k in kwargs for k in ["mindim", "maxdim", "cutoff"]):
        # See https://github.com/QunaSys/quri-parts/pull/203#discussion_r1329458816
        error = np.nan
        psi = jl.normalize(psi)
    exp: float = jl.expectation(psi, op)

    return _Estimate(value=exp, error=error)


def create_itensor_mps_estimator(
    *,
    maxdim: Optional[int] = None,
    cutoff: Optional[float] = None,
    **kwargs: Any,
) -> QuantumEstimator[ITensorStateT]:
    """Returns a :class:`~QuantumEstimator` that uses ITensor MPS simulator to
    calculate expectation values.

    The following parameters including keyword
    arguments `**kwargs` are passed to `ITensors.apply
    <https://itensor.github.io/ITensors.jl/dev/MPSandMPO.html#ITensors.product-Tuple{ITensor,%20ITensors.AbstractMPS}>`_.

    Args:
        maxdim: The maximum number of singular values.
        cutoff: Singular value truncation cutoff.
    """
    ensure_itensor_loaded()

    def estimator(operator: Estimatable, state: ITensorStateT) -> Estimate[complex]:
        if maxdim is not None:
            kwargs["maxdim"] = maxdim
        if cutoff is not None:
            kwargs["cutoff"] = cutoff
        return _estimate(operator, state, **kwargs)

    return estimator


def _sequential_estimate_single_state(
    state: ITensorStateT,
    operators: Sequence[Estimatable],
    **kwargs: Any,
) -> Sequence[Estimate[complex]]:
    qubits = state.qubit_count
    s: juliacall.VectorValue = jl.siteinds("Qubit", qubits)
    psi: juliacall.AnyValue = jl.init_state(s, qubits)
    circuit = convert_circuit(state.circuit, s)
    psi = jl.apply(circuit, psi, **kwargs)
    if any(k in kwargs for k in ["mindim", "maxdim", "cutoff"]):
        psi = jl.normalize(psi)

    results = []
    for op in operators:
        if op == zero():
            results.append(_Estimate(value=0.0, error=0.0))
            continue
        itensor_op = convert_operator(op, s)

        # See https://github.com/QunaSys/quri-parts/pull/203#discussion_r1329458816
        error = 0.0
        if any(k in kwargs for k in ["mindim", "maxdim", "cutoff"]):
            error = np.nan

        results.append(_Estimate(value=jl.expectation(psi, itensor_op), error=error))
    return results


def _concurrent_estimate(
    _sequential_estimate: Callable[
        [Any, Sequence[tuple[Estimatable, ITensorStateT]]], Sequence[Estimate[complex]]
    ],
    _sequential_estimate_single_state: Callable[
        [ITensorStateT, Sequence[Estimatable]], Sequence[Estimate[complex]]
    ],
    operators: Collection[Estimatable],
    states: Collection[ITensorStateT],
    executor: Optional["Executor"],
    concurrency: int = 1,
) -> Sequence[Estimate[complex]]:
    num_ops = len(operators)
    num_states = len(states)

    if num_ops == 0:
        raise ValueError("No operator specified.")

    if num_states == 0:
        raise ValueError("No state specified.")

    if num_ops > 1 and num_states > 1 and num_ops != num_states:
        raise ValueError(
            f"Number of operators ({num_ops}) does not match"
            f"number of states ({num_states})."
        )

    if num_states == 1:
        return execute_concurrently(
            _sequential_estimate_single_state,
            next(iter(states)),
            operators,
            executor,
            concurrency,
        )
    else:
        if num_ops == 1:
            operators = [next(iter(operators))] * num_states
        return execute_concurrently(
            _sequential_estimate, None, zip(operators, states), executor, concurrency
        )


def create_itensor_mps_concurrent_estimator(
    executor: Optional["Executor"] = None,
    concurrency: int = 1,
    *,
    maxdim: Optional[int] = None,
    cutoff: Optional[float] = None,
    **kwargs: Any,
) -> ConcurrentQuantumEstimator[ITensorStateT]:
    """Returns a :class:`~ConcurrentQuantumEstimator` that uses ITensor MPS
    simulator to calculate expectation values.

    For now, this function works when the executor is defined like below

    Examples:
        >>> with ProcessPoolExecutor(
                max_workers=2, mp_context=get_context("spawn")
            ) as executor:

    The following parameters including
    keyword arguments `**kwargs` are passed to `ITensors.apply
    <https://itensor.github.io/ITensors.jl/dev/MPSandMPO.html#ITensors.product-Tuple{ITensor,%20ITensors.AbstractMPS}>`_.

    Args:
        maxdim: The maximum number of singular values.
        cutoff: Singular value truncation cutoff.
    """
    ensure_itensor_loaded()

    if maxdim is not None:
        kwargs["maxdim"] = maxdim
    if cutoff is not None:
        kwargs["cutoff"] = cutoff

    mps_estimator = create_itensor_mps_estimator(**kwargs)

    def _estimate_sequentially(
        _: Any, op_state_tuples: Sequence[tuple[Estimatable, ITensorStateT]]
    ) -> Sequence[Estimate[complex]]:
        return [mps_estimator(operator, state) for operator, state in op_state_tuples]

    def _estimate_single_state_sequentially(
        state: ITensorStateT, operators: Sequence[Estimatable]
    ) -> Sequence[Estimate[complex]]:
        return _sequential_estimate_single_state(state, operators, **kwargs)

    def estimator(
        operators: Collection[Estimatable],
        states: Collection[ITensorStateT],
    ) -> Iterable[Estimate[complex]]:
        return _concurrent_estimate(
            _estimate_sequentially,
            _estimate_single_state_sequentially,
            operators,
            states,
            executor,
            concurrency,
        )

    return estimator


def _sequential_parametric_estimate(
    op_state: tuple[Estimatable, ITensorParametricStateT],
    params: Sequence[Sequence[float]],
    **kwargs: Any,
) -> Sequence[Estimate[complex]]:
    operator, state = op_state
    estimates = []
    estimator = create_itensor_mps_parametric_estimator(**kwargs)
    for param in params:
        estimates.append(estimator(operator, state, param))
    return estimates


def create_itensor_mps_parametric_estimator(
    *,
    maxdim: Optional[int] = None,
    cutoff: Optional[float] = None,
    **kwargs: Any,
) -> ParametricQuantumEstimator[ITensorParametricStateT]:
    """Creates parametric estimator that uses ITensor MPS simulator to
    calculate expectation values.

    The following parameters including
    keyword arguments `**kwargs` are passed to `ITensors.apply
    <https://itensor.github.io/ITensors.jl/dev/MPSandMPO.html#ITensors.product-Tuple{ITensor,%20ITensors.AbstractMPS}>`_.

    Args:
        maxdim: The maximum number of singular values.
        cutoff: Singular value truncation cutoff.
    """
    ensure_itensor_loaded()

    if maxdim is not None:
        kwargs["maxdim"] = maxdim
    if cutoff is not None:
        kwargs["cutoff"] = cutoff

    return create_parametric_estimator(create_itensor_mps_estimator(**kwargs))


def create_itensor_mps_concurrent_parametric_estimator(
    executor: Optional["Executor"] = None,
    concurrency: int = 1,
    *,
    maxdim: Optional[int] = None,
    cutoff: Optional[float] = None,
    **kwargs: Any,
) -> ConcurrentParametricQuantumEstimator[ITensorParametricStateT]:
    """Creates concurrent parametric estimator from parametric estimator.

    The following parameters including
    keyword arguments `**kwargs` are passed to `ITensors.apply
    <https://itensor.github.io/ITensors.jl/dev/MPSandMPO.html#ITensors.product-Tuple{ITensor,%20ITensors.AbstractMPS}>`_.

    Args:
        maxdim: The maximum number of singular values.
        cutoff: Singular value truncation cutoff.
    """
    ensure_itensor_loaded()

    if maxdim is not None:
        kwargs["maxdim"] = maxdim
    if cutoff is not None:
        kwargs["cutoff"] = cutoff

    def _estimate_sequentially(
        op_state: tuple[Estimatable, ITensorParametricStateT],
        params: Sequence[Sequence[float]],
    ) -> Sequence[Estimate[complex]]:
        return _sequential_parametric_estimate(
            op_state,
            params,
            **kwargs,
        )

    def estimator(
        operator: Estimatable,
        state: ITensorParametricStateT,
        params: Sequence[Sequence[float]],
    ) -> Sequence[Estimate[complex]]:
        return execute_concurrently(
            _estimate_sequentially,
            (operator, state),
            params,
            executor,
            concurrency,
        )

    return estimator
