# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

import juliacall
from juliacall import Main as jl

from quri_parts.core.operator import PAULI_IDENTITY, Operator, pauli_label
from quri_parts.itensor.operator import convert_operator

abs_dir = os.path.dirname(os.path.abspath(__file__))
library_path = os.path.join(abs_dir, "../../../quri_parts/itensor/library.jl")
include_statement = 'include("' + library_path + '")'
jl.seval(include_statement)
library_path = os.path.join(abs_dir, "operator_test_library.jl")
jl.seval('include("' + library_path + '")')

jl.seval("using ITensors")


def test_convert_pauli_label() -> None:
    s: juliacall.VectorValue = jl.siteinds("Qubit", 6)
    pauli = pauli_label("Z0 Z2 Z5")
    itensor_op = convert_operator(pauli, s)
    psi: juliacall.AnyValue = jl.init_state(s)
    exp: float = jl.expectation(psi, itensor_op)
    assert exp == -1.0


def test_convert_operator() -> None:
    s: juliacall.VectorValue = jl.siteinds("Qubit", 6)
    op = Operator(
        {
            pauli_label("Z0 Z2 Z5"): 0.1,
            PAULI_IDENTITY: 0.5j,
        }
    )
    itensor_op = convert_operator(op, s)
    psi: juliacall.AnyValue = jl.init_state(s)
    exp: float = jl.expectation(psi, itensor_op)
    assert exp == -0.1 + 0.5j
