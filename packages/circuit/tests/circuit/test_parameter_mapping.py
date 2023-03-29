# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Callable

import pytest

from quri_parts.circuit import CONST, Parameter
from quri_parts.circuit.parameter_mapping import (
    LinearParameterMapping,
    ParameterMappingBase,
    ParameterOrLinearFunction,
    ParameterValueAssignment,
)


class TestParameterMappingBase:
    def test_seq_mapper(self) -> None:
        in_p0 = Parameter("in0")
        in_p1 = Parameter("in1")
        out_p0 = Parameter("out0")
        out_p1 = Parameter("out1")

        def mock_mapper(vals: ParameterValueAssignment) -> ParameterValueAssignment:
            return {
                out_p0: 0.1 * vals[in_p0] + 0.2 * vals[in_p1] + 0.3,
                out_p1: 0.4 * vals[in_p0] + 0.5 * vals[in_p1] + 0.6,
            }

        @dataclass
        class M(ParameterMappingBase):
            in_params: Sequence[Parameter] = (in_p0, in_p1)
            out_params: Sequence[Parameter] = (out_p0, out_p1)
            mapper: Callable[
                [ParameterValueAssignment], ParameterValueAssignment
            ] = mock_mapper

            def get_mapper_element(
                self, out_param: Parameter
            ) -> Callable[[ParameterValueAssignment], float]:
                def m(vals: ParameterValueAssignment) -> float:
                    return mock_mapper(vals)[out_param]

                return m

            @property
            def is_trivial_mapping(self) -> bool:
                return False

        mapping = M()
        seq_mapper = mapping.seq_mapper

        assert seq_mapper((0.2, 0.5)) == (
            0.1 * 0.2 + 0.2 * 0.5 + 0.3,
            0.4 * 0.2 + 0.5 * 0.5 + 0.6,
        )

        with pytest.raises(ValueError):
            seq_mapper((0.2, 0.5, 0.4))


class TestLinearParameterMapping:
    def test_mapper(self) -> None:
        in_p0 = Parameter("in0")
        in_p1 = Parameter("in1")
        out_p0 = Parameter("out0")
        out_p1 = Parameter("out1")
        m: Mapping[Parameter, ParameterOrLinearFunction] = {
            out_p0: {in_p0: 0.1, in_p1: 0.2, CONST: 0.3},
            out_p1: in_p0,
        }

        mapping = LinearParameterMapping(
            in_params=(in_p0, in_p1), out_params=(out_p0, out_p1), mapping=m
        )
        mapper = mapping.mapper

        assert not mapping.is_trivial_mapping
        assert mapper({in_p0: 0.2, in_p1: 0.5}) == {
            out_p0: 0.1 * 0.2 + 0.2 * 0.5 + 0.3,
            out_p1: 0.2,
        }

    def test_get_drivatives(self) -> None:
        in_p0 = Parameter("in0")
        in_p1 = Parameter("in1")
        out_p0 = Parameter("out0")
        out_p1 = Parameter("out1")
        m: Mapping[Parameter, ParameterOrLinearFunction] = {
            out_p0: {in_p0: 0.1, in_p1: 0.2, CONST: 0.3},
            out_p1: in_p0,
        }

        mapping = LinearParameterMapping(
            in_params=(in_p0, in_p1), out_params=(out_p0, out_p1), mapping=m
        )

        derivatives = mapping.get_derivatives()

        derivative0 = derivatives[0]
        assert derivative0.in_params == mapping.in_params
        assert derivative0.out_params == mapping.out_params
        assert derivative0.mapping == {
            out_p0: {CONST: 0.1},
            out_p1: {CONST: 1.0},
        }

        derivative1 = derivatives[1]
        assert derivative1.in_params == mapping.in_params
        assert derivative1.out_params == mapping.out_params
        assert derivative1.mapping == {
            out_p0: {CONST: 0.2},
        }

    def test_trivial_mapping(self) -> None:
        in_p0 = Parameter("in0")
        in_p1 = Parameter("in1")
        out_p0 = Parameter("out0")
        out_p1 = Parameter("out1")
        m: Mapping[Parameter, ParameterOrLinearFunction] = {
            out_p0: {in_p1: 1.0},
            out_p1: in_p0,
        }
        mapping = LinearParameterMapping(
            in_params=(in_p0, in_p1), out_params=(out_p0, out_p1), mapping=m
        )

        assert mapping.is_trivial_mapping
