# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from abc import abstractproperty
from collections.abc import Iterable, Sequence
from typing import Callable, Optional, Protocol, TypeVar, Union, cast, overload

from typing_extensions import TypeAlias

from quri_parts.core.operator import Operator, PauliLabel
from quri_parts.core.state import (
    CircuitQuantumState,
    ParametricCircuitQuantumState,
    ParametricQuantumStateVector,
    QuantumStateVector,
)

EstimateValue = TypeVar("EstimateValue", float, complex, covariant=True)


class Estimate(Protocol[EstimateValue]):
    """Estimate is an interface for classes representing an estimate for a
    certain quantity.

    This interface only contains read-only properties, so an
    implementation can be a (frozen) dataclass or a namedtuple.
    """

    @abstractproperty
    def value(self) -> EstimateValue:
        """The estimate (estimated value) itself."""
        ...

    @abstractproperty
    def error(self) -> float:
        """Represents the \"error\" of the estimate.

        The precise meaning of the \"error\" depends on what type the
        estimate is. If the estimate is a sample mean calculated by
        sampling from some sample distribution, the error is the
        standard error calculated from the samples. If the estimate is
        an exact value calculated without sampling, the error is zero.
        """
        ...


class Estimates(Protocol[EstimateValue]):
    """Estimates is an interface for classes representing estimates for a
    certain quantity.

    This interface only contains read-only properties, so an
    implementation can be a (frozen) dataclass or a namedtuple.
    """

    @abstractproperty
    def values(self) -> Sequence[EstimateValue]:
        """The estimates (estimated values) themselves."""
        ...

    @abstractproperty
    def error_matrix(self) -> Optional[Sequence[Sequence[float]]]:
        """Represents the \"error\" of estimate values.

        The precise meaning of the \"error\" depends on what type the
        estimate is. Basically, if we can get N estimate values, this
        will return N x N covariance matrix.
        """
        ...


class MatrixEstimates(Protocol[EstimateValue]):
    """MatrixEstimates is an interface for classes representing an N x N matrix
    estimate for a certain quantity.

    This interface only contains read-only properties, so an
    implementation can be a (frozen) dataclass or a namedtuple.
    """

    @abstractproperty
    def values(self) -> Sequence[Sequence[EstimateValue]]:
        """The estimates (estimated values) themselves."""
        ...

    @abstractproperty
    def error_tensor(self) -> Optional[Sequence[Sequence[Sequence[Sequence[float]]]]]:
        """Represents the \"error\" of estimate values.

        The precise meaning of the \"error\" depends on what type the
        estimate is. Basically, if we can get N x N estimate values,
        this will return N x N x N x N error tensor.
        """
        ...


#: (Concurrent)QuantumEstimator accepts a single :class:`~PauliLabel` as well as an
#: :class:`~Operator`. Here we call them an "Estimatable".
Estimatable: TypeAlias = Union[Operator, PauliLabel]

#: A type variable represents *any* non-parametric quantum state classes.
#: This is different from :class:`quri_parts.core.state.QuantumStateT`;
#: ``QuantumStateT`` represents *either one of* the classes, while ``_StateT`` also
#: covers *a union of* multiple state classes.
_StateT = TypeVar("_StateT", bound=Union[CircuitQuantumState, QuantumStateVector])

#: QuantumEstimator represents a function that estimates an expectation value of
#: a given :class:`~Operator` for a given non-parametric state.
#: It theoretically corresponds to a value given by sandwiching the operator between
#: a bra and a ket of the state.
#: This is a generic type and you need to specify what kind of state classes
#: it is applicable to.
QuantumEstimator: TypeAlias = Callable[[Estimatable, _StateT], Estimate[complex]]

#: ConcurrentQuantumEstimator represents a function that estimates expectation values of
#: given :class:`~Operator`\ s for given non-parametric states.
#: It basically works in the same way as :class:`~QuantumEstimator`, except that
#: it performs estimation for multiple operators and states concurrently.
#: Numbers of operators and states (i.e. lengths of the first and second arguments)
#: should satisfy one of the followings:
#:
#: * Only one operator is specified.
#: * Only one state is specified.
#: * The number of the operators is the same as the number of the states. In this case,
#:   an operator and a state with the same index are used to estimate one expectation
#:   value.
#:
#: This is a generic type and you need to specify what kind of state classes
#: it is applicable to.
ConcurrentQuantumEstimator: TypeAlias = Callable[
    [Sequence[Estimatable], Sequence[_StateT]],
    Iterable[Estimate[complex]],
]

#: A type variable represents *any* parametric quantum state classes.
#: This is different from :class:`quri_parts.core.state.ParametricQuantumStateT`;
#: ``ParametricQuantumStateT`` represents *either one of* the classes,
#: while ``_ParametricStateT`` also covers *a union of* multiple state classes.
_ParametricStateT = TypeVar(
    "_ParametricStateT",
    bound=Union[ParametricCircuitQuantumState, ParametricQuantumStateVector],
)

#: ParametricQuantumEstimator represents a function that estimates an expectation value
#: of a given :class:`~Operator` for a given parametric state
#: with given parameter values (the third argument).
#: This is a generic type and you need to specify what kind of state classes
#: it is applicable to.
ParametricQuantumEstimator: TypeAlias = Callable[
    [Estimatable, _ParametricStateT, Sequence[float]], Estimate[complex]
]

#: ConcurrentParametricQuantumEstimator represents a function that estimates
# expectation values of a given :class:`~Operator` for a given
# parametric state with multiple sets of parameter values
# (the third argument). It basically works in the same way as
# :class:`~ParametricQuantumEstimator`, except that it performs estimation for multiple
# sets of parameter values concurrently. Length of the returned iterable is the same as
# the number of given parameter sets.
#: This is a generic type and you need to specify what kind of state classes
#: it is applicable to.
ConcurrentParametricQuantumEstimator: TypeAlias = Callable[
    [Estimatable, _ParametricStateT, Sequence[Sequence[float]]],
    Iterable[Estimate[complex]],
]


@overload
def create_parametric_estimator(
    estimator: QuantumEstimator[Union[CircuitQuantumState, QuantumStateVector]],
) -> ParametricQuantumEstimator[
    Union[ParametricCircuitQuantumState, ParametricQuantumStateVector]
]:
    ...


@overload
def create_parametric_estimator(
    estimator: QuantumEstimator[CircuitQuantumState],
) -> ParametricQuantumEstimator[ParametricCircuitQuantumState]:
    ...


@overload
def create_parametric_estimator(
    estimator: QuantumEstimator[QuantumStateVector],
) -> ParametricQuantumEstimator[ParametricQuantumStateVector]:
    ...


def create_parametric_estimator(
    estimator: QuantumEstimator[_StateT],
) -> ParametricQuantumEstimator[_ParametricStateT]:
    """Creates parametric estimator from estimator."""

    def parametric_estimator(
        operator: Estimatable,
        state: _ParametricStateT,
        params: Sequence[float],
    ) -> Estimate[complex]:
        s = cast(_StateT, state.bind_parameters(params))
        return estimator(operator, s)

    return parametric_estimator


def create_concurrent_parametric_estimator(
    parametric_estimator: ParametricQuantumEstimator[_ParametricStateT],
) -> ConcurrentParametricQuantumEstimator[_ParametricStateT]:
    """Creates concurrent parametric estimator from parametric estimator."""

    def concurrent_parametric_estimator(
        operator: Estimatable,
        state: _ParametricStateT,
        seq_of_params: Sequence[Sequence[float]],
    ) -> list[Estimate[complex]]:
        return [
            parametric_estimator(operator, state, params) for params in seq_of_params
        ]

    return concurrent_parametric_estimator


#: GradientEstimator represents a function that estimates gradient values
#: of an expectation value of a given :class:`~Operator` for a given parametric state
#: with given parameter values (the third argument).
#: This is a generic type and you need to specify what kind of state classes
#: it is applicable to.
GradientEstimator: TypeAlias = Callable[
    [Estimatable, _ParametricStateT, Sequence[float]],
    Estimates[complex],
]


#: HessianEstimator represents a function that estimates hessian values
#: of an expectation value of a given :class:`~Operator` for a given parametric state
#: with given parameter values (the third argument).
#: This is a generic type and you need to specify what kind of state classes
#: it is applicable to.
HessianEstimator: TypeAlias = Callable[
    [Estimatable, _ParametricStateT, Sequence[float]], MatrixEstimates[complex]
]

#: OverlapEstimator represents a function that estimates the magnitude squared overlap
#: of two non-parametric quantum states. This should be used when the magnitude of the
#: inner product between two quantum states is needed. It should be symmetric in the
#: input arguments.
#: This is a generic type and you need to specify what kind of state classes
#: it is applicable to.
OverlapEstimator: TypeAlias = Callable[[_StateT, _StateT], Estimate[float]]


#: OverlapWeightedSumEstimator represents a function that estimates the magnitude
#: squared overlaps of two sets of states and produces a weighted sum. It must be
#: passed three :class:`~Sequence`s with the same length. This can be used to
#: evaluate overlap based penalty terms in a Hamiltonian as is done with e.g.
#: VQD. The output should be invariant under permutation of the first and second
#: input arguments.
#: This is a generic type and you need to specify what kind of state classes
#: it is applicable to.
OverlapWeightedSumEstimator: TypeAlias = Callable[
    [Sequence[_StateT], Sequence[_StateT], Sequence[complex]], Estimate[complex]
]


#: ParametricOverlapWeightedSumEstimator represents a function that estimates the
#: magnitude squared overlap of a set of parametric states and returns their
#: weighted sum. This is intended for use in variational algorithms where a
#: parametrized overlap penalty term is needed.
#: The output should be invariant under permutation of the first and second
#: input arguments.
#: This is a generic type and you need to specify what kind of state classes
#: it is applicable to.
ParametricOverlapWeightedSumEstimator: TypeAlias = Callable[
    [
        tuple[_ParametricStateT, Sequence[Sequence[float]]],
        tuple[_ParametricStateT, Sequence[Sequence[float]]],
        Sequence[complex],
    ],
    Estimate[complex],
]


@overload
def create_parametric_overlap_weighted_sum_estimator(
    estimator: OverlapWeightedSumEstimator[
        Union[CircuitQuantumState, QuantumStateVector]
    ]
) -> ParametricOverlapWeightedSumEstimator[
    Union[ParametricCircuitQuantumState, ParametricQuantumStateVector]
]:
    ...


@overload
def create_parametric_overlap_weighted_sum_estimator(
    estimator: OverlapWeightedSumEstimator[QuantumStateVector],
) -> ParametricOverlapWeightedSumEstimator[ParametricQuantumStateVector]:
    ...


@overload
def create_parametric_overlap_weighted_sum_estimator(
    estimator: OverlapWeightedSumEstimator[CircuitQuantumState],
) -> ParametricOverlapWeightedSumEstimator[ParametricCircuitQuantumState]:
    ...


def create_parametric_overlap_weighted_sum_estimator(
    estimator: OverlapWeightedSumEstimator[_StateT],
) -> ParametricOverlapWeightedSumEstimator[_ParametricStateT]:
    def parametric_estimator(
        kets: tuple[_ParametricStateT, Sequence[Sequence[float]]],
        bras: tuple[_ParametricStateT, Sequence[Sequence[float]]],
        weights: Sequence[complex],
    ) -> Estimate[complex]:
        bound_kets = [
            cast(_StateT, kets[0].bind_parameters(params)) for params in kets[1]
        ]
        bound_bras = [
            cast(_StateT, bras[0].bind_parameters(params)) for params in bras[1]
        ]

        return estimator(bound_kets, bound_bras, weights)

    return parametric_estimator
