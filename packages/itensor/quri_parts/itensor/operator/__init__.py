# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections.abc import Iterable
from typing import Union

import juliacall
from juliacall import Main as jl

from quri_parts.core.operator import Operator, PauliLabel, pauli_name


def convert_operator(
    operator: Union[Operator, PauliLabel], qubit_sites: juliacall.VectorValue
) -> juliacall.AnyValue:
    """Convert an :class:`~Operator` or a :class:`~PauliLabel` to an ITensor
    MPO operator.

    qubit_sites: collection of N "Qubit" sites. please follow
    `the Itensor doc <https://itensor.github.io/ITensors.jl/
    stable/IncludedSiteTypes.html#%22Qubit%22-SiteType>`_
    """
    paulis: Iterable[tuple[PauliLabel, complex]]
    if isinstance(operator, Operator):
        paulis = operator.items()
    else:
        paulis = [(operator, 1)]
    os: juliacall.AnyValue = jl.OpSum()
    for pauli, coef in paulis:
        pauli_ops: juliacall.VectorValue = jl.gate_list()
        if len(pauli) == 0:
            os = jl.add_coef_identity_to_opsum(os, coef)
            continue
        for i, p in pauli:
            pauli_ops = jl.add_pauli_to_pauli_product(pauli_ops, pauli_name(p), i + 1)
        os = jl.add_pauli_product_to_opsum(os, coef, pauli_ops)
    op: juliacall.AnyValue = jl.MPO(os, qubit_sites)
    return op
