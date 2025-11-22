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

from quri_parts.circuit.parameter import CONST, Parameter
from quri_parts.circuit.parameter_mapping import LinearParameterMapping
from quri_parts.circuit.parameter_shift import ParameterShiftsAndCoef, ShiftedParameters


class TestShiftedParameters:
    def test_get_derivatives(self) -> None:
        in_p0 = Parameter("in0")
        in_p1 = Parameter("in1")
        out_p0 = Parameter("out0")
        out_p1 = Parameter("out1")

        mapping = LinearParameterMapping(
            in_params=(in_p0, in_p1),
            out_params=(out_p0, out_p1),
            mapping={
                out_p0: {in_p0: 0.1, in_p1: 0.2, CONST: 0.3},
                out_p1: in_p0,
            },
        )
        parameter_shift = ShiftedParameters(mapping)
        derivatives = parameter_shift.get_derivatives()

        assert len(derivatives) == 2

        # derivative w.r.t. in_p0
        derivative0 = derivatives[0]
        assert derivative0.param_mapping == mapping
        assert derivative0.shifts_with_coef == {
            ParameterShiftsAndCoef(shifts=frozenset({(out_p0, 1)}), coef=0.1 / 2),
            ParameterShiftsAndCoef(shifts=frozenset({(out_p0, -1)}), coef=-0.1 / 2),
            ParameterShiftsAndCoef(shifts=frozenset({(out_p1, 1)}), coef=1.0 / 2),
            ParameterShiftsAndCoef(shifts=frozenset({(out_p1, -1)}), coef=-1.0 / 2),
        }

        # derivative w.r.t. in_p1
        derivative1 = derivatives[1]
        assert derivative1.param_mapping == mapping
        assert derivative1.shifts_with_coef == {
            ParameterShiftsAndCoef(shifts=frozenset({(out_p0, 1)}), coef=0.2 / 2),
            ParameterShiftsAndCoef(shifts=frozenset({(out_p0, -1)}), coef=-0.2 / 2),
        }

        derivatives1 = derivative1.get_derivatives()

        assert len(derivatives1) == 2

        # derivative w.r.t. in_p1, in_p0
        derivative10 = derivatives1[0]
        assert derivative10.param_mapping == mapping
        assert derivative10.shifts_with_coef == {
            ParameterShiftsAndCoef(
                shifts=frozenset({(out_p0, 2)}), coef=(0.2 / 2) * 0.1 / 2
            ),
            ParameterShiftsAndCoef(shifts=frozenset(), coef=(-0.2 / 2) * 0.1 / 2 * 2),
            ParameterShiftsAndCoef(
                shifts=frozenset({(out_p0, -2)}), coef=(0.2 / 2) * 0.1 / 2
            ),
            ParameterShiftsAndCoef(
                shifts=frozenset({(out_p0, 1), (out_p1, 1)}), coef=(0.2 / 2) * 1.0 / 2
            ),
            ParameterShiftsAndCoef(
                shifts=frozenset({(out_p0, 1), (out_p1, -1)}),
                coef=(0.2 / 2) * (-1.0) / 2,
            ),
            ParameterShiftsAndCoef(
                shifts=frozenset({(out_p0, -1), (out_p1, 1)}), coef=(-0.2 / 2) * 1.0 / 2
            ),
            ParameterShiftsAndCoef(
                shifts=frozenset({(out_p0, -1), (out_p1, -1)}),
                coef=(-0.2 / 2) * (-1.0) / 2,
            ),
        }

    def test_get_shifted_parameters_and_coef(self) -> None:
        in_p0 = Parameter("in0")
        in_p1 = Parameter("in1")
        out_p0 = Parameter("out0")
        out_p1 = Parameter("out1")
        mapping = LinearParameterMapping(
            in_params=(in_p0, in_p1),
            out_params=(out_p0, out_p1),
            mapping={
                out_p0: {in_p0: 0.1, in_p1: 0.2, CONST: 0.3},
                out_p1: in_p0,
            },
        )
        shifts_with_coef = frozenset(
            {
                ParameterShiftsAndCoef(shifts=frozenset({(out_p0, 1)}), coef=0.1 / 2),
                ParameterShiftsAndCoef(shifts=frozenset({(out_p0, -1)}), coef=-0.1 / 2),
                ParameterShiftsAndCoef(shifts=frozenset({(out_p1, 1)}), coef=1.0 / 2),
                ParameterShiftsAndCoef(shifts=frozenset({(out_p1, -1)}), coef=-1.0 / 2),
                ParameterShiftsAndCoef(shifts=frozenset(), coef=-0.5 / 2),
            }
        )
        parameter_shift = ShiftedParameters(
            param_mapping=mapping, shifts_with_coef=shifts_with_coef
        )

        out0, out1 = (0.1 * 0.9 + 0.2 * 0.7 + 0.3, 0.9)
        params_and_coef = parameter_shift.get_shifted_parameters_and_coef((0.9, 0.7))
        assert set(params_and_coef) == {
            ((out0 + 1 * math.pi / 2, out1), 0.1 / 2),
            ((out0 - 1 * math.pi / 2, out1), -0.1 / 2),
            ((out0, out1 + 1 * math.pi / 2), 1.0 / 2),
            ((out0, out1 - 1 * math.pi / 2), -1.0 / 2),
            ((out0, out1), -0.5 / 2),
        }
