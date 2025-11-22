# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from openfermion import FermionOperator as OpenFermionFermionOperator
from openfermion import hermitian_conjugated as hermitian_conjugated_openfermion


class FermionOperator(OpenFermionFermionOperator):  # type: ignore
    """Wrapper class for OpenFermion's :class:`FermionOperator`."""

    def hermitian_conjugated(self) -> "FermionOperator":
        """Returns the hermitian conjugate of itself."""
        openfermion_op = hermitian_conjugated_openfermion(self)
        return fermion_operator_from_openfermion_op(openfermion_op)


def fermion_operator_from_openfermion_op(
    openfermion_op: OpenFermionFermionOperator,
) -> FermionOperator:
    """Create :class:`FermionOperator` from OpenFermion's
    :class:`FermionOperator`"""
    op = FermionOperator()
    for pauli_product, coef in openfermion_op.terms.items():
        op += FermionOperator(
            pauli_product,
            coef,
        )
    return op


# ported from openfermion.ops.operators.fermion_operator.py
def has_particle_number_symmetry(
    f_op: "OpenFermionFermionOperator", check_spin_symmetry: bool = False
) -> bool:
    """Query whether operator conserves particle number and (optionally) spin
    symmetry.

    Args:
        f_op: fermionic operator (FermionOperator)
        check_spin_symmetry: boolean value
    Returns:
        bool
    """

    for term in f_op.terms:
        spin = 0
        particles = 0
        for operator in term:
            particles += (-1) ** (operator[1] + 1)  # add 1 if create, else -1
            spin += (-1) ** (operator[0] + operator[1])
        if particles:
            return False
        elif spin and check_spin_symmetry:
            return False
    return True
