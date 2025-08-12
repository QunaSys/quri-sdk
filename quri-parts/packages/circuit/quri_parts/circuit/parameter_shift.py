# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from typing import NamedTuple, cast

from typing_extensions import TypeAlias

from .parameter import CONST, Parameter
from .parameter_mapping import LinearParameterFunction, LinearParameterMapping

#: Represents a set of parameter shifts appearing in the parameter shift rule.
#: The shift for each parameter is stored as an integer. The actual shift is calculated
#: by multiplying the integer with ``math.pi / 2``.
ParameterShifts: TypeAlias = frozenset[tuple[Parameter, int]]


class ParameterShiftsAndCoef(NamedTuple):
    """A pair of parameter shifts and a coefficient appearing in the parameter
    shift rule."""

    #: Shifts for parmeters.
    shifts: ParameterShifts
    #: A differential coefficient associated with the parameter shifts.
    coef: float


NO_SHIFT = ParameterShiftsAndCoef(frozenset(), 1.0)


def _get_linear_deriv(deriv: LinearParameterMapping, param: Parameter) -> float:
    if param not in deriv.mapping:
        return 0.0
    fn = cast(LinearParameterFunction, deriv.mapping[param])
    return fn[CONST]


@dataclass(frozen=True)
class ShiftedParameters:
    """A set of parameter shifts and associated coefficients obtained as a
    result of applying parameter shift rule.

    Note:
        The current implementation support only linear mappings between input and
        output parameters (i.e. :class:`~LinearParameterMapping`).

    When created without ``shifts_with_coef``, it corresponds to the original
    parametric circuit (i.e. only one component with no parameter shift and coefficient
    = 1.0).
    Instances returned by :meth:`get_derivatives` method contains pairs of parameter
    shifts and associated coefficients. To evaluate derivative of an expectation value
    of an observable, estimate an expectation value of the observable with parameters
    obtained by :meth:`get_shifted_parameters_and_coef`, multiply it with the
    coefficient, and then sum up all terms.
    """

    param_mapping: LinearParameterMapping
    shifts_with_coef: Collection[ParameterShiftsAndCoef] = (NO_SHIFT,)

    def _get_derivative(
        self, deriv_mapping: LinearParameterMapping
    ) -> "ShiftedParameters":
        new_shifts_map: dict[ParameterShifts, float] = {}
        for shifts, coef in self.shifts_with_coef:
            for raw_p in self.param_mapping.out_params:
                c = _get_linear_deriv(deriv_mapping, raw_p)
                if c == 0.0:
                    continue
                for sign in (1, -1):
                    s = dict(shifts)
                    new_shift = s.get(raw_p, 0) + sign
                    if new_shift == 0:
                        del s[raw_p]
                    else:
                        s[raw_p] = new_shift
                    key = frozenset(s.items())
                    new_shifts_map[key] = (
                        new_shifts_map.get(key, 0) + coef * c * sign / 2.0
                    )
        new_shifts = frozenset(
            {ParameterShiftsAndCoef(k, v) for k, v in new_shifts_map.items()}
        )
        return ShiftedParameters(
            self.param_mapping,
            new_shifts,
        )

    def get_derivatives(self) -> Sequence["ShiftedParameters"]:
        r"""Returns a sequence of :class:`~ShiftedParameters`\ s corresponding
        to derivatives of the original :class:`~ShiftedParameters` with respect
        to each input parameter.

        When the original instance corresponds to the original parametric circuit with
        a linear mapping on :math:`\theta^\text{(in)}_i`, given an arbitrary observable
        :math:`\hat{O}`, the returned sequence corresponds to derivatives

        .. math::
            \left(
                \frac{\partial\langle\hat{O}\rangle}{\partial\theta^\text{(in)}_0},
                \ldots,
                \frac{\partial\langle\hat{O}\rangle}{\partial\theta^\text{(in)}_{m-1}},
            \right).
        """
        deriv_mappings = self.param_mapping.get_derivatives()
        return [self._get_derivative(deriv_mapping) for deriv_mapping in deriv_mappings]

    def get_shifted_parameters_and_coef(
        self, param_vals: Sequence[float]
    ) -> Collection[tuple[Sequence[float], float]]:
        """Returns shifted parameter values and associated coefficients, which
        can be used for evaluation of derivatives, given parameter values where
        the derivatives are evaluated."""
        out_param_vals = self.param_mapping.mapper(
            dict(zip(self.param_mapping.in_params, param_vals))
        )
        result = []
        for shifts, coef in self.shifts_with_coef:
            d = dict(out_param_vals)
            for out_p, shift in shifts:
                d[out_p] = d[out_p] + shift * math.pi / 2
            shifted_vars = tuple(d[out_p] for out_p in self.param_mapping.out_params)
            result.append((shifted_vars, coef))
        return result
