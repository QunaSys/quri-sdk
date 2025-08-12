# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
from openfermion import FermionOperator as OpenFermionFermionOperator
from openfermion import (
    symmetry_conserving_bravyi_kitaev as of_symmetry_conserving_bravyi_kitaev,
)

from quri_parts.core.operator import PAULI_IDENTITY, Operator, pauli_label
from quri_parts.core.state import ComputationalBasisState
from quri_parts.openfermion.operator import (
    FermionOperator,
    operator_from_openfermion_op,
)
from quri_parts.openfermion.transforms import (
    bravyi_kitaev,
    jordan_wigner,
    symmetry_conserving_bravyi_kitaev,
)


class TestOperatorMapper:
    def test_scbk_op_mapper(self) -> None:
        n_spin_orbitals = 4
        n_fermions = 2

        const = FermionOperator("")
        f_op1 = FermionOperator("3^ 1")
        f_op2 = FermionOperator("2^ 0")
        f_op3 = FermionOperator("0^")
        f_op4 = FermionOperator("1^ 0")
        op_total = const + f_op1 + f_op2 + f_op3 + f_op4

        of_const = OpenFermionFermionOperator("")
        of_f_op1 = OpenFermionFermionOperator("3^ 1")
        of_f_op2 = OpenFermionFermionOperator("2^ 0")
        of_f_op3 = OpenFermionFermionOperator("0^")
        of_f_op4 = OpenFermionFermionOperator("1^ 0")
        of_op_total = of_const + of_f_op1 + of_f_op2 + of_f_op3 + of_f_op4

        of_op_symmetry_conserve = of_const + of_f_op1 + of_f_op2

        op_mapper = symmetry_conserving_bravyi_kitaev.get_of_operator_mapper(
            n_spin_orbitals, n_fermions, 0.0
        )
        total_transformed = op_mapper(op_total)
        of_total_transformed = op_mapper(of_op_total)

        expected = operator_from_openfermion_op(
            of_symmetry_conserving_bravyi_kitaev(of_op_symmetry_conserve, 4, 2)
        )

        assert total_transformed == expected
        assert of_total_transformed == expected

        op_mapper = symmetry_conserving_bravyi_kitaev(
            n_spin_orbitals, n_fermions, 0.0
        ).of_operator_mapper
        total_transformed = op_mapper(op_total)
        of_total_transformed = op_mapper(of_op_total)
        assert total_transformed == expected
        assert of_total_transformed == expected

        # n_spin_up = 2, n_fermions = 2
        n_spin_orbitals = 4
        n_fermions = 2
        sz = 1.0

        expected = Operator({PAULI_IDENTITY: 1})
        expected.add_term(pauli_label("X0"), 0.25)  # (0.25+0j) [X0]
        expected.add_term(pauli_label("X0"), (+1) * -0.25)  # (-0.25+0j) [X0 Z1]
        expected.add_term(pauli_label("Y0"), 0.25j)  # 0.25j [Y0]
        expected.add_term(pauli_label("Y0"), (+1) * -0.25j)  # -0.25j [Y0 Z1]
        expected.add_term(pauli_label("X1"), (+1) * -0.25)  # (-0.25+0j) [Z1 X2 Z3]
        expected.add_term(pauli_label("Y1"), (+1) * -0.25j)  # -0.25j [Z1 Y2 Z3]
        expected.add_term(pauli_label("X1"), 0.25)  # (0.25+0j) [X2]
        expected.add_term(pauli_label("Y1"), 0.25j)  # 0.25j [Y2]

        op_mapper = symmetry_conserving_bravyi_kitaev(
            n_spin_orbitals, n_fermions, sz
        ).of_operator_mapper
        assert op_mapper(op_total) == expected

        # n_spin_up = 1, n_fermions = 3
        n_spin_orbitals = 4
        n_fermions = 3
        sz = -0.5

        expected = Operator({PAULI_IDENTITY: 1})
        expected.add_term(pauli_label("X0"), 0.25)  # (0.25+0j) [X0]
        expected.add_term(pauli_label("X0"), (-1) * -0.25)  # (-0.25+0j) [X0 Z1]
        expected.add_term(pauli_label("Y0"), 0.25j)  # 0.25j [Y0]
        expected.add_term(pauli_label("Y0"), (-1) * -0.25j)  # -0.25j [Y0 Z1]
        expected.add_term(pauli_label("X1"), (+1) * -0.25)  # (-0.25+0j) [Z1 X2 Z3]
        expected.add_term(pauli_label("Y1"), (+1) * -0.25j)  # -0.25j [Z1 Y2 Z3]
        expected.add_term(pauli_label("X1"), 0.25)  # (0.25+0j) [X2]
        expected.add_term(pauli_label("Y1"), 0.25j)  # 0.25j [Y2]

        op_mapper = symmetry_conserving_bravyi_kitaev(
            n_spin_orbitals, n_fermions, sz
        ).of_operator_mapper
        assert op_mapper(op_total) == expected

        # n_spin_up = 2, n_fermions = 3
        n_spin_orbitals = 4
        n_fermions = 3
        sz = 0.5

        expected = Operator({PAULI_IDENTITY: 1})
        expected.add_term(pauli_label("X0"), 0.25)  # (0.25+0j) [X0]
        expected.add_term(pauli_label("X0"), (+1) * -0.25)  # (-0.25+0j) [X0 Z1]
        expected.add_term(pauli_label("Y0"), 0.25j)  # 0.25j [Y0]
        expected.add_term(pauli_label("Y0"), (+1) * -0.25j)  # -0.25j [Y0 Z1]
        expected.add_term(pauli_label("X1"), (-1) * -0.25)  # (-0.25+0j) [Z1 X2 Z3]
        expected.add_term(pauli_label("Y1"), (-1) * -0.25j)  # -0.25j [Z1 Y2 Z3]
        expected.add_term(pauli_label("X1"), 0.25)  # (0.25+0j) [X2]
        expected.add_term(pauli_label("Y1"), 0.25j)  # 0.25j [Y2]

        op_mapper = symmetry_conserving_bravyi_kitaev(
            n_spin_orbitals, n_fermions, sz
        ).of_operator_mapper
        assert op_mapper(op_total) == expected

        expected = operator_from_openfermion_op(
            of_symmetry_conserving_bravyi_kitaev(
                of_op_symmetry_conserve, n_spin_orbitals, n_fermions
            )
        )
        assert op_mapper(op_total) == expected


class TestStateMapper:
    def test_jw_state_mapper(self) -> None:
        n_spin_orbitals = 4
        state_mapper = jordan_wigner.get_state_mapper(n_spin_orbitals)
        inv_state_mapper = jordan_wigner.get_inv_state_mapper(n_spin_orbitals)

        mapped = state_mapper([0, 1])
        assert mapped == ComputationalBasisState(n_spin_orbitals, bits=0b0011)

        inv_mapped = inv_state_mapper(
            ComputationalBasisState(n_spin_orbitals, bits=0b0011)
        )
        assert inv_mapped == [0, 1]

    def test_jw_state_mapper_property(self) -> None:
        n_spin_orbitals = 4
        state_mapper = jordan_wigner(n_spin_orbitals).state_mapper
        inv_state_mapper = jordan_wigner(n_spin_orbitals).inv_state_mapper

        mapped = state_mapper([0, 1])
        assert mapped == ComputationalBasisState(n_spin_orbitals, bits=0b0011)

        inv_mapped = inv_state_mapper(
            ComputationalBasisState(n_spin_orbitals, bits=0b0011)
        )
        assert inv_mapped == [0, 1]

    def test_bk_state_mapper(self) -> None:
        n_spin_orbitals = 4
        state_mapper = bravyi_kitaev.get_state_mapper(n_spin_orbitals)
        inv_state_mapper = bravyi_kitaev.get_inv_state_mapper(n_spin_orbitals)

        # State transformation matrix for BK(n_spin_orbitals=4, n_fermions=2):
        # [[1, 0, 0, 0]
        #  [1, 1, 0, 0]
        #  [0, 0, 1, 0]
        #  [1, 1, 1, 1]]

        mapped = state_mapper([0, 1])
        assert mapped == ComputationalBasisState(n_spin_orbitals, bits=0b0001)

        inv_mapped = inv_state_mapper(
            ComputationalBasisState(n_spin_orbitals, bits=0b0001)
        )
        assert inv_mapped == [0, 1]

        mapped = state_mapper([0, 2])
        assert mapped == ComputationalBasisState(n_spin_orbitals, bits=0b0111)

        inv_mapped = inv_state_mapper(
            ComputationalBasisState(n_spin_orbitals, bits=0b0111)
        )
        assert inv_mapped == [0, 2]

        n_spin_orbitals = 8
        n_fermions = 4
        state_mapper = bravyi_kitaev.get_state_mapper(n_spin_orbitals, n_fermions)
        inv_state_mapper = bravyi_kitaev.get_inv_state_mapper(
            n_spin_orbitals, n_fermions
        )

        # State transformation matrix for BK(n_spin_orbitals=8, n_fermions=4):
        # [[1, 0, 0, 0, 0, 0, 0, 0]
        #  [1, 1, 0, 0, 0, 0, 0, 0]
        #  [0, 0, 1, 0, 0, 0, 0, 0]
        #  [1, 1, 1, 1, 0, 0, 0, 0]
        #  [0, 0, 0, 0, 1, 0, 0, 0]
        #  [0, 0, 0, 0, 1, 1, 0, 0]
        #  [0, 0, 0, 0, 0, 0, 1, 0]
        #  [1, 1, 1, 1, 1, 1, 1, 1]]

        mapped = state_mapper([0, 1, 2, 3])
        assert mapped == ComputationalBasisState(n_spin_orbitals, bits=0b00000101)

        inv_mapped = inv_state_mapper(
            ComputationalBasisState(n_spin_orbitals, bits=0b00000101)
        )
        assert inv_mapped == [0, 1, 2, 3]

        mapped = state_mapper([0, 1, 4, 5])
        assert mapped == ComputationalBasisState(n_spin_orbitals, bits=0b00010001)

        inv_mapped = inv_state_mapper(
            ComputationalBasisState(n_spin_orbitals, bits=0b00010001)
        )
        assert inv_mapped == [0, 1, 4, 5]

    def test_bk_state_mapper_property(self) -> None:
        n_spin_orbitals = 4
        state_mapper = bravyi_kitaev(n_spin_orbitals).state_mapper
        inv_state_mapper = bravyi_kitaev(n_spin_orbitals).inv_state_mapper
        # State transformation matrix for BK(n_spin_orbitals=4, n_fermions=2):
        # [[1, 0, 0, 0]
        #  [1, 1, 0, 0]
        #  [0, 0, 1, 0]
        #  [1, 1, 1, 1]]

        mapped = state_mapper([0, 1])
        assert mapped == ComputationalBasisState(n_spin_orbitals, bits=0b0001)

        inv_mapped = inv_state_mapper(
            ComputationalBasisState(n_spin_orbitals, bits=0b0001)
        )
        assert inv_mapped == [0, 1]

        mapped = state_mapper([0, 2])
        assert mapped == ComputationalBasisState(n_spin_orbitals, bits=0b0111)

        inv_mapped = inv_state_mapper(
            ComputationalBasisState(n_spin_orbitals, bits=0b0111)
        )
        assert inv_mapped == [0, 2]

        n_spin_orbitals = 8
        n_fermions = 4
        state_mapper = bravyi_kitaev(n_spin_orbitals, n_fermions).state_mapper
        inv_state_mapper = bravyi_kitaev(n_spin_orbitals, n_fermions).inv_state_mapper

        # State transformation matrix for BK(n_spin_orbitals=8, n_fermions=4):
        # [[1, 0, 0, 0, 0, 0, 0, 0]
        #  [1, 1, 0, 0, 0, 0, 0, 0]
        #  [0, 0, 1, 0, 0, 0, 0, 0]
        #  [1, 1, 1, 1, 0, 0, 0, 0]
        #  [0, 0, 0, 0, 1, 0, 0, 0]
        #  [0, 0, 0, 0, 1, 1, 0, 0]
        #  [0, 0, 0, 0, 0, 0, 1, 0]
        #  [1, 1, 1, 1, 1, 1, 1, 1]]

        mapped = state_mapper([0, 1, 2, 3])
        assert mapped == ComputationalBasisState(n_spin_orbitals, bits=0b00000101)

        inv_mapped = inv_state_mapper(
            ComputationalBasisState(n_spin_orbitals, bits=0b00000101)
        )
        assert inv_mapped == [0, 1, 2, 3]

        mapped = state_mapper([0, 1, 4, 5])
        assert mapped == ComputationalBasisState(n_spin_orbitals, bits=0b00010001)

        inv_mapped = inv_state_mapper(
            ComputationalBasisState(n_spin_orbitals, bits=0b00010001)
        )
        assert inv_mapped == [0, 1, 4, 5]

    def test_scbk_state_mapper(self) -> None:
        n_spin_orbitals = 4

        with pytest.raises(ValueError):
            symmetry_conserving_bravyi_kitaev.get_state_mapper(n_spin_orbitals)

        n_fermions = 2
        state_mapper = symmetry_conserving_bravyi_kitaev.get_state_mapper(
            n_spin_orbitals, n_fermions, 0.0
        )
        inv_state_mapper = symmetry_conserving_bravyi_kitaev.get_inv_state_mapper(
            n_spin_orbitals, n_fermions, 0.0
        )

        # State transformation:
        # 1. Reorder to "all spin-up orbitals, then all spin-down orbitals"
        # 2. Apply the matrix for BK(n_spin_orbitals=4, n_fermions=2):
        # [[1, 0, 0, 0]
        #  [1, 1, 0, 0]
        #  [0, 0, 1, 0]
        #  [1, 1, 1, 1]]
        # 3. Drop 2nd and 4th qubits

        mapped = state_mapper([0, 1])
        assert mapped == ComputationalBasisState(n_spin_orbitals - 2, bits=0b11)
        inv_mapped = inv_state_mapper(
            ComputationalBasisState(n_spin_orbitals - 2, bits=0b11)
        )
        assert inv_mapped == [0, 1]

        mapped = state_mapper([0, 3])
        assert mapped == ComputationalBasisState(n_spin_orbitals - 2, bits=0b01)
        inv_mapped = inv_state_mapper(
            ComputationalBasisState(n_spin_orbitals - 2, bits=0b01)
        )
        assert inv_mapped == [0, 3]

        mapped = state_mapper([1, 2])
        assert mapped == ComputationalBasisState(n_spin_orbitals - 2, bits=0b10)
        inv_mapped = inv_state_mapper(
            ComputationalBasisState(n_spin_orbitals - 2, bits=0b10)
        )
        assert inv_mapped == [1, 2]

        mapped = state_mapper([2, 3])
        assert mapped == ComputationalBasisState(n_spin_orbitals - 2, bits=0b00)
        inv_mapped = inv_state_mapper(
            ComputationalBasisState(n_spin_orbitals - 2, bits=0b00)
        )
        assert inv_mapped == [2, 3]

        n_spin_orbitals = 8
        n_fermions = 4
        state_mapper = symmetry_conserving_bravyi_kitaev.get_state_mapper(
            n_spin_orbitals, n_fermions, 0.0
        )
        inv_state_mapper = symmetry_conserving_bravyi_kitaev.get_inv_state_mapper(
            n_spin_orbitals, n_fermions, 0.0
        )

        # State transformation:
        # 1. Reorder to "all spin-up orbitals, then all spin-down orbitals"
        # 2. Apply the matrix for BK(n_spin_orbitals=8, n_fermions=4):
        # [[1, 0, 0, 0, 0, 0, 0, 0]
        #  [1, 1, 0, 0, 0, 0, 0, 0]
        #  [0, 0, 1, 0, 0, 0, 0, 0]
        #  [1, 1, 1, 1, 0, 0, 0, 0]
        #  [0, 0, 0, 0, 1, 0, 0, 0]
        #  [0, 0, 0, 0, 1, 1, 0, 0]
        #  [0, 0, 0, 0, 0, 0, 1, 0]
        #  [1, 1, 1, 1, 1, 1, 1, 1]]
        # 3. Drop 4th and 8th qubits

        mapped = state_mapper([0, 1, 2, 3])
        assert mapped == ComputationalBasisState(n_spin_orbitals - 2, bits=0b001001)

        inv_mapped = inv_state_mapper(
            ComputationalBasisState(n_spin_orbitals - 2, bits=0b001001)
        )
        assert inv_mapped == [0, 1, 2, 3]

        # In this case, since qubit binary before dropping 2 qubits is larger than
        # 2**n_qubits-1, it is required to take bitwise-AND when creating the state
        mapped = state_mapper([0, 1, 4, 5])
        assert mapped == ComputationalBasisState(n_spin_orbitals - 2, bits=0b111111)

        inv_mapped = inv_state_mapper(
            ComputationalBasisState(n_spin_orbitals - 2, bits=0b111111)
        )
        assert inv_mapped == [0, 1, 4, 5]

    def test_scbk_state_mapper_property(self) -> None:
        n_spin_orbitals = 4

        with pytest.raises(ValueError):
            symmetry_conserving_bravyi_kitaev(n_spin_orbitals).state_mapper

        n_fermions = 2
        state_mapper = symmetry_conserving_bravyi_kitaev(
            n_spin_orbitals, n_fermions, 0.0
        ).state_mapper
        inv_state_mapper = symmetry_conserving_bravyi_kitaev(
            n_spin_orbitals, n_fermions, 0.0
        ).inv_state_mapper

        # State transformation:
        # 1. Reorder to "all spin-up orbitals, then all spin-down orbitals"
        # 2. Apply the matrix for BK(n_spin_orbitals=4, n_fermions=2):
        # [[1, 0, 0, 0]
        #  [1, 1, 0, 0]
        #  [0, 0, 1, 0]
        #  [1, 1, 1, 1]]
        # 3. Drop 2nd and 4th qubits

        mapped = state_mapper([0, 1])
        assert mapped == ComputationalBasisState(n_spin_orbitals - 2, bits=0b11)
        inv_mapped = inv_state_mapper(
            ComputationalBasisState(n_spin_orbitals - 2, bits=0b11)
        )
        assert inv_mapped == [0, 1]

        mapped = state_mapper([0, 3])
        assert mapped == ComputationalBasisState(n_spin_orbitals - 2, bits=0b01)
        inv_mapped = inv_state_mapper(
            ComputationalBasisState(n_spin_orbitals - 2, bits=0b01)
        )
        assert inv_mapped == [0, 3]

        mapped = state_mapper([1, 2])
        assert mapped == ComputationalBasisState(n_spin_orbitals - 2, bits=0b10)
        inv_mapped = inv_state_mapper(
            ComputationalBasisState(n_spin_orbitals - 2, bits=0b10)
        )
        assert inv_mapped == [1, 2]

        mapped = state_mapper([2, 3])
        assert mapped == ComputationalBasisState(n_spin_orbitals - 2, bits=0b00)
        inv_mapped = inv_state_mapper(
            ComputationalBasisState(n_spin_orbitals - 2, bits=0b00)
        )
        assert inv_mapped == [2, 3]

        n_spin_orbitals = 8
        n_fermions = 4
        state_mapper = symmetry_conserving_bravyi_kitaev(
            n_spin_orbitals, n_fermions, 0.0
        ).state_mapper
        inv_state_mapper = symmetry_conserving_bravyi_kitaev(
            n_spin_orbitals, n_fermions, 0.0
        ).inv_state_mapper

        # State transformation:
        # 1. Reorder to "all spin-up orbitals, then all spin-down orbitals"
        # 2. Apply the matrix for BK(n_spin_orbitals=8, n_fermions=4):
        # [[1, 0, 0, 0, 0, 0, 0, 0]
        #  [1, 1, 0, 0, 0, 0, 0, 0]
        #  [0, 0, 1, 0, 0, 0, 0, 0]
        #  [1, 1, 1, 1, 0, 0, 0, 0]
        #  [0, 0, 0, 0, 1, 0, 0, 0]
        #  [0, 0, 0, 0, 1, 1, 0, 0]
        #  [0, 0, 0, 0, 0, 0, 1, 0]
        #  [1, 1, 1, 1, 1, 1, 1, 1]]
        # 3. Drop 4th and 8th qubits

        mapped = state_mapper([0, 1, 2, 3])
        assert mapped == ComputationalBasisState(n_spin_orbitals - 2, bits=0b001001)

        inv_mapped = inv_state_mapper(
            ComputationalBasisState(n_spin_orbitals - 2, bits=0b001001)
        )
        assert inv_mapped == [0, 1, 2, 3]

        # In this case, since qubit binary before dropping 2 qubits is larger than
        # 2**n_qubits-1, it is required to take bitwise-AND when creating the state
        mapped = state_mapper([0, 1, 4, 5])
        assert mapped == ComputationalBasisState(n_spin_orbitals - 2, bits=0b111111)

        inv_mapped = inv_state_mapper(
            ComputationalBasisState(n_spin_orbitals - 2, bits=0b111111)
        )
        assert inv_mapped == [0, 1, 4, 5]

        # Test for generic spin SCBK: integer spin
        n_spin_orbitals = 8
        n_fermions = 4
        state_mapper = symmetry_conserving_bravyi_kitaev(
            n_spin_orbitals, n_fermions, 1.0
        ).state_mapper
        inv_state_mapper = symmetry_conserving_bravyi_kitaev(
            n_spin_orbitals, n_fermions, 1.0
        ).inv_state_mapper

        mapped = state_mapper([0, 1, 2, 4])
        assert mapped == ComputationalBasisState(n_spin_orbitals - 2, bits=0b11101)

        inv_mapped = inv_state_mapper(
            ComputationalBasisState(n_spin_orbitals - 2, bits=0b11101)
        )
        assert inv_mapped == [0, 1, 2, 4]

        mapped = state_mapper([0, 2, 3, 4])
        assert mapped == ComputationalBasisState(n_spin_orbitals - 2, bits=0b10101)

        inv_mapped = inv_state_mapper(
            ComputationalBasisState(n_spin_orbitals - 2, bits=0b10101)
        )
        assert inv_mapped == [0, 2, 3, 4]

        mapped = state_mapper([0, 1, 2, 6])
        assert mapped == ComputationalBasisState(n_spin_orbitals - 2, bits=0b11001)

        inv_mapped = inv_state_mapper(
            ComputationalBasisState(n_spin_orbitals - 2, bits=0b11001)
        )
        assert inv_mapped == [0, 1, 2, 6]

        # Test for generic spin SCBK: half integer spin
        n_spin_orbitals = 8
        n_fermions = 3
        state_mapper = symmetry_conserving_bravyi_kitaev(
            n_spin_orbitals, n_fermions, -1.5
        ).state_mapper
        inv_state_mapper = symmetry_conserving_bravyi_kitaev(
            n_spin_orbitals, n_fermions, -1.5
        ).inv_state_mapper

        mapped = state_mapper([1, 3, 5])
        assert mapped == ComputationalBasisState(n_spin_orbitals - 2, bits=0b101000)

        inv_mapped = inv_state_mapper(
            ComputationalBasisState(n_spin_orbitals - 2, bits=0b101000)
        )
        assert inv_mapped == [1, 3, 5]

        mapped = state_mapper([1, 3, 7])
        assert mapped == ComputationalBasisState(n_spin_orbitals - 2, bits=0b1000)

        inv_mapped = inv_state_mapper(
            ComputationalBasisState(n_spin_orbitals - 2, bits=0b1000)
        )
        assert inv_mapped == [1, 3, 7]

        mapped = state_mapper([3, 5, 7])
        assert mapped == ComputationalBasisState(n_spin_orbitals - 2, bits=0b110000)

        inv_mapped = inv_state_mapper(
            ComputationalBasisState(n_spin_orbitals - 2, bits=0b110000)
        )
        assert inv_mapped == [3, 5, 7]
