# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from abc import ABC, abstractproperty
from collections import defaultdict
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Callable, Protocol, Union, cast

from typing_extensions import TypeAlias

from .circuit_parametric import CONST, Parameter

#: ParameterValueAssignment represents a :class:`~Mapping` which assigns concrete
#: values to a set of parameters.
ParameterValueAssignment: TypeAlias = Mapping[Parameter, float]
#: Mapper represents a function that maps a set of parameter values (input) to another
#: set of parameter values (output).
Mapper: TypeAlias = Callable[[ParameterValueAssignment], ParameterValueAssignment]
#: SeqMapper represents a function that maps a sequence of parameter values (input) to
#: another sequence of parameter values (output).
SeqMapper: TypeAlias = Callable[[Sequence[float]], Sequence[float]]


class ParameterMapping(Protocol):
    r"""ParameterMapping represents a mapping of a set of parameters to another
    set of parameters.

    It can be considered as a function mapping :math:`m` input parameters to :math:`n`
    output parameters:

    .. math::
        (\theta^\text{(out)}_0, \ldots, \theta^\text{(out)}_{n-1}) =
        f(\theta^\text{(in)}_0, \ldots, \theta^\text{(in)}_{m-1})
    """

    @abstractproperty
    def in_params(self) -> Sequence[Parameter]:
        r"""Returns a sequence of input :class:`~Parameter`\ s."""
        ...

    @abstractproperty
    def out_params(self) -> Sequence[Parameter]:
        r"""Returns a sequence of output :class:`~Parameter`\ s."""
        ...

    @abstractproperty
    def mapper(self) -> Mapper:
        """Returns a function that maps an input
        :class:`~ParameterValueAssignment` to an output
        :class:`~ParameterValueAssignment` (i.e. :math:`f`).

        It is expected that the function captures the state of
        :class:`~ParameterMapping` at the time that this property is accessed and does
        not reflect subsequent changes in the state of the :class:`~ParameterMapping`
        instance.
        """
        ...

    @abstractproperty
    def seq_mapper(self) -> SeqMapper:
        """Returns a function that maps a sequence of input parameter values to
        a sequence of output parameter values (i.e. :math:`f`).

        It is expected that the function captures the state of
        :class:`~ParameterMapping` at the time that this property is accessed and does
        not reflect subsequent changes in the state of the :class:`~ParameterMapping`
        instance.
        """
        ...

    def get_derivatives(self) -> Sequence["ParameterMapping"]:
        r"""Returns a sequence of :class:`~ParameterMapping`\ s of derivatives
        of the original :class:`~ParameterMapping` with respect to each input
        parameter.

        The returned sequence corresponds to

        .. math::
            \left(
                \frac{\partial f}{\partial \theta^\text{(in)}_0},
                \ldots,
                \frac{\partial f}{\partial \theta^\text{(in)}_{m-1}}
            \right),

        where :math:`f` is defined as described in :class:`~ParameterMapping` docstring.
        """
        raise NotImplementedError


class ParameterMappingBase(ABC):
    """An abstract class used as a base for implementing
    :class:`ParameterMapping`.

    Currently this class only has a general implementation of
    :meth:`seq_mapper` method.
    """

    @abstractproperty
    def in_params(self) -> Sequence[Parameter]:
        ...

    @abstractproperty
    def out_params(self) -> Sequence[Parameter]:
        ...

    @abstractproperty
    def mapper(self) -> Mapper:
        ...

    @abstractproperty
    def is_trivial_mapping(self) -> bool:
        """Returns if the mapping is trivial one-to-one mapping (Identity
        function)."""

    @property
    def seq_mapper(self) -> SeqMapper:
        in_params = self.in_params
        out_params = self.out_params
        mapper = self.mapper

        def m(param_vals: Sequence[float]) -> Sequence[float]:
            if len(param_vals) != len(in_params):
                raise ValueError(
                    f"Passed value count ({len(param_vals)}) does not match parameter "
                    f"count ({len(in_params)})."
                )
            d = mapper(dict(zip(in_params, param_vals)))
            return tuple(d[p] for p in out_params)

        return m


#: A type representing a linear (affine) function of parameters. It is an alias for
#: mapping from :class:`~Parameter` to float coefficients. A constant term can be
#: represented as a coefficient for ``CONST``.
LinearParameterFunction: TypeAlias = Mapping[Parameter, float]
#: A union type of :class:`~Parameter` and :class:`~LinearParameterFunction`.
ParameterOrLinearFunction: TypeAlias = Union[Parameter, LinearParameterFunction]


class LinearParameterMapping(ParameterMappingBase):
    r"""A :class:`~ParameterMapping` representing a linear (affine)
    transformation of parameters.

    The mapping is represented as a :class:`Mapping` from :class:`Parameter`\ s to
    :class:`ParameterOrLinearFunction`\ s. For example, if the mapping is defined as:

    .. math::
        \begin{align}
            \theta^\text{(out)}_0 &= 0.1\theta^\text{(in)}_1 + 0.2, \\
            \theta^\text{(out)}_1 &= 0.3\theta^\text{(in)}_0 + 0.4\theta^\text{(in)}_1,
        \end{align}

    the ``mapping`` argument should be as the following:

    .. highlight:: python
    .. code-block:: python

        {
            out_param_0: { in_param_1: 0.1, CONST: 0.2 },
            out_param_1: { in_param_0: 0.3, in_param_1: 0.4 },
        }

    where ``in_param_0(1)`` and ``out_param_0(1)`` are input and output
    :class:`~Parameter` instances respectively.
    """

    def __init__(
        self,
        in_params: Sequence[Parameter] = (),
        out_params: Sequence[Parameter] = (),
        mapping: Mapping[Parameter, ParameterOrLinearFunction] = {},
    ):
        self._in_params: Sequence[Parameter] = tuple(in_params)
        self._out_params: Sequence[Parameter] = tuple(out_params)
        self._mapping: Mapping[Parameter, ParameterOrLinearFunction] = _freeze_map(
            mapping
        )

    @staticmethod
    def _from_immutable_data(
        in_params: Sequence[Parameter],
        out_params: Sequence[Parameter],
        mapping: Mapping[Parameter, ParameterOrLinearFunction],
    ) -> "LinearParameterMapping":
        m = LinearParameterMapping()
        m._in_params = in_params
        m._out_params = out_params
        m._mapping = mapping
        return m

    def with_data_updated(
        self,
        *,
        in_params_addition: Sequence[Parameter] = (),
        out_params_addition: Sequence[Parameter] = (),
        mapping_update: Mapping[Parameter, ParameterOrLinearFunction] = {},
    ) -> "LinearParameterMapping":
        return LinearParameterMapping._from_immutable_data(
            (*self._in_params, *in_params_addition),
            (*self._out_params, *out_params_addition),
            {**self._mapping, **_freeze_map(mapping_update)},
        )

    @property
    def in_params(self) -> Sequence[Parameter]:
        return self._in_params

    @property
    def out_params(self) -> Sequence[Parameter]:
        return self._out_params

    @property
    def mapping(self) -> Mapping[Parameter, ParameterOrLinearFunction]:
        return self._mapping

    @property
    def mapper(self) -> Mapper:
        out_params = self._out_params
        mapping = self._mapping

        def m(param_vals: ParameterValueAssignment) -> ParameterValueAssignment:
            in_param_vals = {**param_vals, CONST: 1.0}
            out_param_vals = {}
            for out_param in out_params:
                fn = mapping[out_param]
                if isinstance(fn, Parameter):
                    v = in_param_vals[fn]
                else:
                    v = sum(c * in_param_vals[p] for p, c in fn.items())
                out_param_vals[out_param] = v
            return out_param_vals

        return m

    @property
    def is_trivial_mapping(self) -> bool:
        if len(self.in_params) != len(self.out_params):
            return False

        used_in_params: list[Parameter] = []
        for out_param in self.out_params:
            mapping = self.mapping[out_param]
            if isinstance(mapping, Parameter):
                if mapping in used_in_params:
                    return False
                used_in_params.append(mapping)
            else:
                mapping = cast(LinearParameterFunction, mapping)
                if (
                    len(mapping) != 1
                    or next(iter(mapping.values())) != 1.0
                    or next(iter(mapping.keys())) in used_in_params
                ):
                    return False
                else:
                    used_in_params.append(next(iter(mapping.keys())))
        return True

    def get_derivatives(self) -> Sequence["LinearParameterMapping"]:
        r"""Returns a sequence of :class:`~LinearParameterMapping`\ s of
        derivatives of the original :class:`~LinearParameterMapping` with
        respect to each input parameter.

        Since the original mapping is linear, the returned mappings only contains
        constant terms. For example, for the linear mapping defined in the docstring of
        :class:`~LinearParameterMapping`, the returned derivatives are as follows:

        .. math::
            \begin{align}
                \frac{\partial f}{\partial \theta^\text{(in)}_0} &= (0, 0.3)\\
                \frac{\partial f}{\partial \theta^\text{(in)}_1} &= (0.1, 0.4)
            \end{align}
        """
        new_mappings: dict[
            Parameter, dict[Parameter, LinearParameterFunction]
        ] = defaultdict(dict)
        for out_param, fn in self.mapping.items():
            if isinstance(fn, Parameter):
                new_mappings[fn][out_param] = {CONST: 1.0}
            else:
                for p, c in fn.items():
                    new_mappings[p][out_param] = {CONST: c}

        in_params = tuple(self.in_params)
        out_params = tuple(self.out_params)
        return [
            LinearParameterMapping._from_immutable_data(
                in_params, out_params, MappingProxyType(new_mappings[p])
            )
            for p in self.in_params
        ]

    def combine(self, other: "LinearParameterMapping") -> "LinearParameterMapping":
        return LinearParameterMapping._from_immutable_data(
            in_params=(*self.in_params, *other.in_params),
            out_params=(*self.out_params, *other.out_params),
            mapping={
                **self.mapping,
                **other.mapping,
            },
        )


def _freeze_map(
    m: Mapping[Parameter, ParameterOrLinearFunction]
) -> Mapping[Parameter, ParameterOrLinearFunction]:
    return MappingProxyType(
        {
            p: fn if isinstance(fn, Parameter) else MappingProxyType(dict(fn))
            for p, fn in m.items()
        }
    )
