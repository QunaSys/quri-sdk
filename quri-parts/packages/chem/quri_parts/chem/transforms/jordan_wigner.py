# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from quri_parts.chem.transforms.fermion_operator import FermionOperator
from quri_parts.core.operator import PAULI_IDENTITY, Operator, pauli_label


def jordan_wigner(operator: FermionOperator) -> Operator:
    r"""Transform FermionOperator to Operator using Jordan-Wigner mapping.

    Jordan-Wigner mapping is given by
    a_j^\dagger -> Z_0 .. Z_{j-1} (X_j - iY_j) / 2
    a_j -> Z_0 .. Z_{j-1} (X_j + iY_j) / 2
    """
    if isinstance(operator, FermionOperator):
        return _jordan_wigner_fermion_operator(operator)
    raise TypeError("Operator must be a FermionOperator, ")


def _jordan_wigner_fermion_operator(operator: FermionOperator) -> Operator:
    transformed_operator = Operator()
    lookup_ladder_terms = dict()
    for label, coef in operator.items():
        # Initialize transformed term by I.
        transformed_term = Operator({PAULI_IDENTITY: coef})
        for ladder_operator in label:
            z_factors = Operator({PAULI_IDENTITY: 1.0})
            if ladder_operator not in lookup_ladder_terms:
                if ladder_operator[0] != 0:
                    z_factors = Operator(
                        {
                            pauli_label(
                                " ".join([f"Z{i}" for i in range(ladder_operator[0])])
                            ): 1.0
                        }
                    )
                pauli_x_component = z_factors * Operator(
                    {pauli_label(f"X{ladder_operator[0]}"): 0.5}
                )
                if ladder_operator[1]:
                    pauli_y_component = z_factors * Operator(
                        {pauli_label(f"Y{ladder_operator[0]}"): -0.5j}
                    )
                else:
                    pauli_y_component = z_factors * Operator(
                        {pauli_label(f"Y{ladder_operator[0]}"): 0.5j}
                    )
                lookup_ladder_terms[ladder_operator] = (
                    pauli_x_component + pauli_y_component
                )
            transformed_term *= lookup_ladder_terms[ladder_operator]
        transformed_operator += transformed_term
    return transformed_operator
