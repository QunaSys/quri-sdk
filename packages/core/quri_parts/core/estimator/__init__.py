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
from dataclasses import dataclass
from typing import Callable, Generic, Optional, Protocol, TypeVar, Union, cast, overload

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


@overload
def create_concurrent_parametric_estimator_from_concurrent_estimator(
    concurrent_estimator: ConcurrentQuantumEstimator[_StateT],
) -> ConcurrentParametricQuantumEstimator[_ParametricStateT]:
    ...


@overload
def create_concurrent_parametric_estimator_from_concurrent_estimator(
    concurrent_estimator: ConcurrentQuantumEstimator[CircuitQuantumState],
) -> ConcurrentParametricQuantumEstimator[ParametricCircuitQuantumState]:
    ...


@overload
def create_concurrent_parametric_estimator_from_concurrent_estimator(
    concurrent_estimator: ConcurrentQuantumEstimator[QuantumStateVector],
) -> ConcurrentParametricQuantumEstimator[ParametricQuantumStateVector]:
    ...


def create_concurrent_parametric_estimator_from_concurrent_estimator(
    concurrent_estimator: ConcurrentQuantumEstimator[_StateT],
) -> ConcurrentParametricQuantumEstimator[_ParametricStateT]:
    """Creates a concurrent parametric estimator from a concurrent
    estimator."""

    def concurrent_parametric_estimator(
        operator: Estimatable,
        state: _ParametricStateT,
        seq_of_params: Sequence[Sequence[float]],
    ) -> Iterable[Estimate[complex]]:
        bound_states = cast(
            Sequence[_StateT], [state.bind_parameters(param) for param in seq_of_params]
        )
        return concurrent_estimator([operator], bound_states)

    return concurrent_parametric_estimator


@overload
def create_parametric_estimator_from_concurrent_estimator(
    concurrent_estimator: ConcurrentQuantumEstimator[_StateT],
) -> ParametricQuantumEstimator[_ParametricStateT]:
    ...


@overload
def create_parametric_estimator_from_concurrent_estimator(
    concurrent_estimator: ConcurrentQuantumEstimator[CircuitQuantumState],
) -> ParametricQuantumEstimator[ParametricCircuitQuantumState]:
    ...


@overload
def create_parametric_estimator_from_concurrent_estimator(
    concurrent_estimator: ConcurrentQuantumEstimator[QuantumStateVector],
) -> ParametricQuantumEstimator[ParametricQuantumStateVector]:
    ...


def create_parametric_estimator_from_concurrent_estimator(
    concurrent_estimator: ConcurrentQuantumEstimator[_StateT],
) -> ParametricQuantumEstimator[_ParametricStateT]:
    """Creates a parametric estimator from a concurrent estimator."""

    def parametric_estimator(
        operator: Estimatable,
        state: _ParametricStateT,
        params: Sequence[float],
    ) -> Estimate[complex]:
        bound_states = cast(_StateT, state.bind_parameters(params))
        estimate = concurrent_estimator([operator], [bound_states])
        return next(iter(estimate))

    return parametric_estimator


def create_estimator_from_concurrent_estimator(
    concurrent_estimator: ConcurrentQuantumEstimator[_StateT],
) -> QuantumEstimator[_StateT]:
    """Creates an estimator from a concurrent estimator."""

    def estimator(
        operator: Estimatable,
        state: _StateT,
    ) -> Estimate[complex]:
        return next(iter(concurrent_estimator([operator], [state])))

    return estimator


def create_concurrent_estimator_from_estimator(
    estimator: QuantumEstimator[_StateT],
) -> ConcurrentQuantumEstimator[_StateT]:
    """Creates a concurrent estimator from an estimator."""

    def concurrent_estimator(
        operators: Sequence[Estimatable],
        states: Sequence[_StateT],
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
            states = [next(iter(states))] * num_ops

        if num_ops == 1:
            operators = [next(iter(operators))] * num_states

        return [estimator(op, state) for op, state in zip(operators, states)]

    return concurrent_estimator


@dataclass
class GeneralQuantumEstimator(Generic[_StateT, _ParametricStateT]):
    r"""A callable dataclass that holds :class:`QuantumEstimator`,
    :class:`ConcurrentQuantumEstimator`, :class:`ParametricQuantumEstimator`,
    or :class:`ConcurrentParametricEstimator`. When it is used as a callable function,
    it allows generic inputs for expectation value estimation. The allowed inputs for
    using it as a callable function are:

    - Act as :class:`QuantumEstimator`:
        - Estimatable, _StateT -> Estimate
    - Act as :class:`ConcurrentQuantumEstimator`:
        - Estimatable, [_StateT, ...] -> [Estimate, ...]
        - [Estimatable], [_StateT, ...] -> [Estimate, ...]
        - [Estimatable, ...], _StateT -> [Estimate, ...]
        - [Estimatable, ...], [_StateT] -> [Estimate, ...]
        - [Estimatable, ...], [_StateT, ...] -> [Estimate, ...]
    - Act as :class:`ParametricQuantumEstimator`:
        - Estimatable, _ParametricStateT, [float, ...] -> Estimate
    - Act as :class:`ConcurrentParametricQuantumEstimator`:
        - Estimatable, _ParametricStateT, [[float, ...], ...] -> [Estimate, ...]

    When a :class:`GeneralQuantumEstimator` is called directly with one of the
    combinations above, it needs to parse the input arguments to figure out which of
    :class:`QuantumEstimator`, :class:`ConcurrentQuantumEstimator`,
    :class:`ParametricQuantumEstimator`, or :class:`ConcurrentParametricEstimator`
    is required to perform the estimation. To avoid such performance penalty, you may
    retrieve the desired estimator as a property directly.
    """

    estimator: QuantumEstimator[_StateT]
    concurrent_estimator: ConcurrentQuantumEstimator[_StateT]
    parametric_estimator: ParametricQuantumEstimator[_ParametricStateT]
    concurrent_parametric_estimator: ConcurrentParametricQuantumEstimator[
        _ParametricStateT
    ]

    @overload
    def __call__(  # type: ignore[overload-overlap]
        self,
        op: Estimatable,
        state: _StateT,
    ) -> Estimate[complex]:
        """A :class:`QuantumEstimator`"""
        ...

    @overload
    def __call__(
        self,
        op: Sequence[Estimatable],
        state: Sequence[_StateT],
    ) -> Iterable[Estimate[complex]]:
        """A :class:`ConcurrentQuantumEstimator`"""
        ...

    @overload
    def __call__(
        self,
        op: Estimatable,
        state: Sequence[_StateT],
    ) -> Iterable[Estimate[complex]]:
        """A :class:`ConcurrentQuantumEstimator`"""
        ...

    @overload
    def __call__(
        self,
        op: Sequence[Estimatable],
        state: _StateT,
    ) -> Iterable[Estimate[complex]]:
        """A :class:`ConcurrentQuantumEstimator`"""
        ...

    @overload
    def __call__(
        self,
        op: Estimatable,
        state: _ParametricStateT,
        param: Iterable[float],
    ) -> Estimate[complex]:
        """A :class:`ParametricQuantumEstimator`"""
        ...

    @overload
    def __call__(
        self,
        op: Estimatable,
        state: _ParametricStateT,
        param: Iterable[Iterable[float]],
    ) -> Iterable[Estimate[complex]]:
        """A :class:`ConcurrentParametricQuantumEstimator`"""
        ...

    def __call__(
        self,
        op: Union[Estimatable, Sequence[Estimatable]],
        state: Union[_StateT, Sequence[_StateT], _ParametricStateT],
        param: Optional[Union[Iterable[float], Iterable[Iterable[float]]]] = None,
    ) -> Union[Estimate[complex], Iterable[Estimate[complex]]]:
        if param is None:
            if isinstance(op, Operator) or isinstance(op, PauliLabel):
                if isinstance(state, Sequence):
                    return self.concurrent_estimator([op], state)
                state = cast(_StateT, state)
                return self.estimator(op, state)

            if isinstance(state, Sequence):
                return self.concurrent_estimator(op, state)
            state = cast(_StateT, state)
            return self.concurrent_estimator(op, [state])

        assert not isinstance(state, Sequence)
        assert isinstance(op, Operator) or isinstance(op, PauliLabel)

        state = cast(_ParametricStateT, state)
        if isinstance(next(iter(param)), Iterable):
            param = cast(Sequence[Sequence[float]], param)
            return self.concurrent_parametric_estimator(op, state, param)
        param = cast(Sequence[float], param)
        return self.parametric_estimator(op, state, param)


@overload
def create_general_estimator_from_estimator(
    estimator: QuantumEstimator[CircuitQuantumState],
) -> GeneralQuantumEstimator[CircuitQuantumState, ParametricCircuitQuantumState]:
    ...


@overload
def create_general_estimator_from_estimator(
    estimator: QuantumEstimator[QuantumStateVector],
) -> GeneralQuantumEstimator[QuantumStateVector, ParametricQuantumStateVector]:
    ...


def create_general_estimator_from_estimator(
    estimator: QuantumEstimator[_StateT],
) -> GeneralQuantumEstimator[_StateT, _ParametricStateT]:
    """Creates a :class:`GeneralQuantumEstimator` from a
    :class:`QuantumEstimator`.

    Note:
    - The concurrencies of the :class:`ConcurrentQuantumEstimaror` and
        `ConcurrentParametricQuantumEstimaror` will be set to 1 when a
        :class:`GeneralQuantumEstimator` is created with this function.
    - When circuit conversion is involved in the estimator execution, the
        parametric estimator created from this function will bind the parameter
        first, and then convert the bound circuit every time the patametric estimator
        is called.
    """
    concurrent_estimator = create_concurrent_estimator_from_estimator(estimator)
    parametric_estimator: ParametricQuantumEstimator[
        _ParametricStateT
    ] = create_parametric_estimator_from_concurrent_estimator(concurrent_estimator)

    concurrent_parametric_estimator: ConcurrentParametricQuantumEstimator[
        _ParametricStateT
    ] = create_concurrent_parametric_estimator_from_concurrent_estimator(
        concurrent_estimator
    )
    general_estimator = GeneralQuantumEstimator(
        estimator,
        concurrent_estimator,
        parametric_estimator,
        concurrent_parametric_estimator,
    )

    return general_estimator


@overload
def create_general_estimator_from_concurrent_estimator(
    concurrent_estimator: ConcurrentQuantumEstimator[CircuitQuantumState],
) -> GeneralQuantumEstimator[CircuitQuantumState, ParametricCircuitQuantumState]:
    ...


@overload
def create_general_estimator_from_concurrent_estimator(
    concurrent_estimator: ConcurrentQuantumEstimator[QuantumStateVector],
) -> GeneralQuantumEstimator[QuantumStateVector, ParametricQuantumStateVector]:
    ...


def create_general_estimator_from_concurrent_estimator(
    concurrent_estimator: ConcurrentQuantumEstimator[_StateT],
) -> GeneralQuantumEstimator[_StateT, _ParametricStateT]:
    """Creates a :class:`GeneralQuantumEstimator` from a
    :class:`ConcurrentQuantumEstimator`.

    Note:
    - When circuit conversion is involved in the estimator execution, the
        parametric estimator created from this function will bind the parameter
        first, and then convert the bound circuit every time the patametric estimator
        is called.
    """
    estimator = create_estimator_from_concurrent_estimator(concurrent_estimator)
    parametric_estimator: ParametricQuantumEstimator[
        _ParametricStateT
    ] = create_parametric_estimator_from_concurrent_estimator(concurrent_estimator)

    concurrent_parametric_estimator: ConcurrentParametricQuantumEstimator[
        _ParametricStateT
    ] = create_concurrent_parametric_estimator_from_concurrent_estimator(
        concurrent_estimator
    )
    general_estimator = GeneralQuantumEstimator(
        estimator,
        concurrent_estimator,
        parametric_estimator,
        concurrent_parametric_estimator,
    )

    return general_estimator
