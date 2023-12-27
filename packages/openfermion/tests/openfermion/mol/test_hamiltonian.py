# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from numpy import array, isclose
from openfermion import FermionOperator

from quri_parts.chem.mol import SpinMO1eIntArray, SpinMO2eIntArray, SpinMOeIntSet, cas
from quri_parts.core.operator import (
    PAULI_IDENTITY,
    Operator,
    pauli_label,
    truncate,
    zero,
)
from quri_parts.openfermion.mol import (
    get_fermionic_hamiltonian,
    get_qubit_mapped_hamiltonian,
    operator_from_of_fermionic_op,
)
from quri_parts.openfermion.transforms import (
    bravyi_kitaev,
    symmetry_conserving_bravyi_kitaev,
)


def test_get_fermionic_hamiltonian() -> None:
    nuc_energy = -1.18702694476004
    spin_mo_1e_int = SpinMO1eIntArray(array([[-0.33696926, 0.0], [0.0, -0.33696926]]))
    spin_mo_2e_int = SpinMO2eIntArray(
        array(
            [
                [[[0.53466412, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.53466412, 0.0]]],
                [[[0.0, 0.53466412], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.53466412]]],
            ]
        )
    )
    spin_mo_eint_set = SpinMOeIntSet(
        const=nuc_energy, mo_1e_int=spin_mo_1e_int, mo_2e_int=spin_mo_2e_int
    )
    fermionic_hamiltonian = get_fermionic_hamiltonian(spin_mo_eint_set)

    expected_hamiltonian = FermionOperator()
    expected_hamiltonian += FermionOperator(((0, 1), (0, 0))) * -0.3369692554860686
    expected_hamiltonian += FermionOperator(((1, 1), (1, 0))) * -0.3369692554860686
    expected_hamiltonian += (
        FermionOperator(((0, 1), (0, 1), (0, 0), (0, 0))) * 0.2673320597646795
    )
    expected_hamiltonian += (
        FermionOperator(((0, 1), (1, 1), (1, 0), (0, 0))) * 0.2673320597646795
    )
    expected_hamiltonian += (
        FermionOperator(((1, 1), (0, 1), (0, 0), (1, 0))) * 0.2673320597646795
    )
    expected_hamiltonian += (
        FermionOperator(((1, 1), (1, 1), (1, 0), (1, 0))) * 0.2673320597646795
    )
    expected_hamiltonian += -1.18702694476004

    assert set(expected_hamiltonian.terms) == set(fermionic_hamiltonian)

    for op, coeff in expected_hamiltonian.terms.items():
        assert isclose(fermionic_hamiltonian[op], coeff)


class TestOperatorFromOfFermionicOp:
    def test_conversion_without_active_space(self) -> None:
        fermionic_operator = FermionOperator()
        fermionic_operator += FermionOperator(((0, 1), (0, 0))) * -1.1108441798837272
        fermionic_operator += FermionOperator(((1, 1), (1, 0))) * -1.1108441798837272
        fermionic_operator += FermionOperator(((2, 1), (2, 0))) * -0.589121003706083
        fermionic_operator += FermionOperator(((3, 1), (3, 0))) * -0.589121003706083
        fermionic_operator += (
            FermionOperator(((0, 1), (0, 1), (0, 0), (0, 0))) * 0.31320124976475894
        )
        fermionic_operator += (
            FermionOperator(((0, 1), (0, 1), (2, 0), (2, 0))) * 0.0983952917427351
        )
        fermionic_operator += (
            FermionOperator(((0, 1), (1, 1), (1, 0), (0, 0))) * 0.31320124976475894
        )
        fermionic_operator += (
            FermionOperator(((0, 1), (1, 1), (3, 0), (2, 0))) * 0.0983952917427351
        )
        fermionic_operator += (
            FermionOperator(((0, 1), (2, 1), (0, 0), (2, 0))) * 0.0983952917427351
        )
        fermionic_operator += (
            FermionOperator(((0, 1), (2, 1), (2, 0), (0, 0))) * 0.31085338155985653
        )
        fermionic_operator += (
            FermionOperator(((0, 1), (3, 1), (1, 0), (2, 0))) * 0.0983952917427351
        )
        fermionic_operator += (
            FermionOperator(((0, 1), (3, 1), (3, 0), (0, 0))) * 0.31085338155985653
        )
        fermionic_operator += (
            FermionOperator(((1, 1), (0, 1), (0, 0), (1, 0))) * 0.31320124976475894
        )
        fermionic_operator += (
            FermionOperator(((1, 1), (0, 1), (2, 0), (3, 0))) * 0.0983952917427351
        )
        fermionic_operator += (
            FermionOperator(((1, 1), (1, 1), (1, 0), (1, 0))) * 0.31320124976475894
        )
        fermionic_operator += (
            FermionOperator(((1, 1), (1, 1), (3, 0), (3, 0))) * 0.0983952917427351
        )
        fermionic_operator += (
            FermionOperator(((1, 1), (2, 1), (0, 0), (3, 0))) * 0.0983952917427351
        )
        fermionic_operator += (
            FermionOperator(((1, 1), (2, 1), (2, 0), (1, 0))) * 0.31085338155985653
        )
        fermionic_operator += (
            FermionOperator(((1, 1), (3, 1), (1, 0), (3, 0))) * 0.0983952917427351
        )
        fermionic_operator += (
            FermionOperator(((1, 1), (3, 1), (3, 0), (1, 0))) * 0.31085338155985653
        )
        fermionic_operator += (
            FermionOperator(((2, 1), (0, 1), (0, 0), (2, 0))) * 0.31085338155985676
        )
        fermionic_operator += (
            FermionOperator(((2, 1), (0, 1), (2, 0), (0, 0))) * 0.0983952917427351
        )
        fermionic_operator += (
            FermionOperator(((2, 1), (1, 1), (1, 0), (2, 0))) * 0.31085338155985676
        )
        fermionic_operator += (
            FermionOperator(((2, 1), (1, 1), (3, 0), (0, 0))) * 0.0983952917427351
        )
        fermionic_operator += (
            FermionOperator(((2, 1), (2, 1), (0, 0), (0, 0))) * 0.0983952917427351
        )
        fermionic_operator += (
            FermionOperator(((2, 1), (2, 1), (2, 0), (2, 0))) * 0.326535373471287
        )
        fermionic_operator += (
            FermionOperator(((2, 1), (3, 1), (1, 0), (0, 0))) * 0.0983952917427351
        )
        fermionic_operator += (
            FermionOperator(((2, 1), (3, 1), (3, 0), (2, 0))) * 0.326535373471287
        )
        fermionic_operator += (
            FermionOperator(((3, 1), (0, 1), (0, 0), (3, 0))) * 0.31085338155985676
        )
        fermionic_operator += (
            FermionOperator(((3, 1), (0, 1), (2, 0), (1, 0))) * 0.0983952917427351
        )
        fermionic_operator += (
            FermionOperator(((3, 1), (1, 1), (1, 0), (3, 0))) * 0.31085338155985676
        )
        fermionic_operator += (
            FermionOperator(((3, 1), (1, 1), (3, 0), (1, 0))) * 0.0983952917427351
        )
        fermionic_operator += (
            FermionOperator(((3, 1), (2, 1), (0, 0), (1, 0))) * 0.0983952917427351
        )
        fermionic_operator += (
            FermionOperator(((3, 1), (2, 1), (2, 0), (3, 0))) * 0.326535373471287
        )
        fermionic_operator += (
            FermionOperator(((3, 1), (3, 1), (1, 0), (1, 0))) * 0.0983952917427351
        )
        fermionic_operator += (
            FermionOperator(((3, 1), (3, 1), (3, 0), (3, 0))) * 0.326535373471287
        )
        fermionic_operator += 0.52917721092

        jw_qubit_operator, _ = operator_from_of_fermionic_op(
            fermionic_operator, cas(2, 2)
        )

        expected_jw_qubit_operator = Operator(
            {
                PAULI_IDENTITY: -0.3276081896748091,
                pauli_label("Z0"): +0.13716572937099503,
                pauli_label("Z1"): +0.13716572937099503,
                pauli_label("Z2"): -0.1303629205710911,
                pauli_label("Z3"): -0.1303629205710911,
                pauli_label("Z0 Z1"): +0.15660062488237947,
                pauli_label("Z0 Z2"): +0.10622904490856078,
                pauli_label("Z0 Z3"): +0.15542669077992832,
                pauli_label("Z1 Z2"): +0.15542669077992832,
                pauli_label("Z1 Z3"): +0.10622904490856078,
                pauli_label("Z2 Z3"): +0.1632676867356435,
                pauli_label("X0 X1 Y2 Y3"): -0.04919764587136755,
                pauli_label("X0 Y1 Y2 X3"): +0.04919764587136755,
                pauli_label("Y0 X1 X2 Y3"): +0.04919764587136755,
                pauli_label("Y0 Y1 X2 X3"): -0.04919764587136755,
            }
        )

        assert truncate(expected_jw_qubit_operator - jw_qubit_operator) == zero()

        bk_qubit_operator, _ = operator_from_of_fermionic_op(
            fermionic_operator,
            cas(2, 2),
            fermion_qubit_mapping=bravyi_kitaev,
        )
        expected_bk_qubit_operator = Operator(
            {
                PAULI_IDENTITY: (-0.32760818967480876 + 0j),
                pauli_label("X0 Z1 X2"): (0.04919764587136755 + 0j),
                pauli_label("X0 Z1 X2 Z3"): (0.04919764587136755 + 0j),
                pauli_label("Y0 Z1 Y2"): (0.04919764587136755 + 0j),
                pauli_label("Y0 Z1 Y2 Z3"): (0.04919764587136755 + 0j),
                pauli_label("Z0"): (0.13716572937099514 + 0j),
                pauli_label("Z0 Z1"): (0.13716572937099503 + 0j),
                pauli_label("Z0 Z1 Z2"): (0.15542669077992832 + 0j),
                pauli_label("Z0 Z1 Z2 Z3"): (0.15542669077992832 + 0j),
                pauli_label("Z0 Z2"): (0.10622904490856078 + 0j),
                pauli_label("Z0 Z2 Z3"): (0.10622904490856078 + 0j),
                pauli_label("Z1"): (0.15660062488237947 + 0j),
                pauli_label("Z1 Z2 Z3"): (-0.1303629205710911 + 0j),
                pauli_label("Z1 Z3"): (0.1632676867356435 + 0j),
                pauli_label("Z2"): (-0.1303629205710911 + 0j),
            }
        )
        assert truncate(expected_bk_qubit_operator - bk_qubit_operator) == zero()

        scbk_qubit_operator, _ = operator_from_of_fermionic_op(
            fermionic_operator,
            cas(2, 2),
            sz=0,
            fermion_qubit_mapping=symmetry_conserving_bravyi_kitaev,
        )
        expected_scbk_qubit_operator = Operator(
            {
                PAULI_IDENTITY: -0.5400662794919303,
                pauli_label("X0 X1"): 0.1967905834854702,
                pauli_label("Z0"): 0.26752864994208625,
                pauli_label("Z0 Z1"): 0.009014930058166337,
                pauli_label("Z1"): 0.26752864994208614,
            }
        )
        assert truncate(scbk_qubit_operator - expected_scbk_qubit_operator) == zero()

    def test_conversion_with_active_space(self) -> None:
        fermionic_operator = FermionOperator()
        fermionic_operator += FermionOperator(((0, 1), (0, 0))) * -0.3369692554860686
        fermionic_operator += FermionOperator(((1, 1), (1, 0))) * -0.3369692554860686
        fermionic_operator += (
            FermionOperator(((0, 1), (0, 1), (0, 0), (0, 0))) * 0.2673320597646795
        )
        fermionic_operator += (
            FermionOperator(((0, 1), (1, 1), (1, 0), (0, 0))) * 0.2673320597646795
        )
        fermionic_operator += (
            FermionOperator(((1, 1), (0, 1), (0, 0), (1, 0))) * 0.2673320597646795
        )
        fermionic_operator += (
            FermionOperator(((1, 1), (1, 1), (1, 0), (1, 0))) * 0.2673320597646795
        )
        fermionic_operator += -1.18702694476004

        jw_qubit_operator, _ = operator_from_of_fermionic_op(
            fermionic_operator, cas(1, 1)
        )
        jw_expected_qubit_operator = Operator(
            {
                PAULI_IDENTITY: -1.3903301703637687,
                pauli_label("Z0"): 0.03481859786069455,
                pauli_label("Z1"): 0.03481859786069455,
                pauli_label("Z0 Z1"): 0.13366602988233975,
            }
        )
        assert truncate(jw_qubit_operator - jw_expected_qubit_operator) == zero()

        bk_qubit_operator, _ = operator_from_of_fermionic_op(
            fermionic_operator, cas(1, 1), fermion_qubit_mapping=bravyi_kitaev
        )
        expected_bk_qubit_operator = Operator(
            {
                PAULI_IDENTITY: -1.3903301747600398,
                pauli_label("Z0"): 0.03481859786069455,
                pauli_label("Z1"): 0.13366602988233975,
                pauli_label("Z0 Z1"): 0.03481859786069455,
            }
        )
        assert truncate(bk_qubit_operator - expected_bk_qubit_operator) == zero()


class TestGetQubitMappedHamiltonianOperator:
    def test_get_qubit_mapped_hamiltonian_operator(self) -> None:
        nuc_energy = 0.52917721092
        spin_mo_1e_int = SpinMO1eIntArray(
            array(
                [
                    [-1.11084418, 0.0, 0.0, 0.0],
                    [0.0, -1.11084418, 0.0, 0.0],
                    [0.0, 0.0, -0.589121, 0.0],
                    [0.0, 0.0, 0.0, -0.589121],
                ]
            )
        )
        spin_mo_2e_int = SpinMO2eIntArray(
            array(
                [
                    [
                        [
                            [0.6264025, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.19679058, 0.0],
                            [0.0, 0.0, 0.0, 0.0],
                        ],
                        [
                            [0.0, 0.0, 0.0, 0.0],
                            [0.6264025, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.19679058, 0.0],
                        ],
                        [
                            [0.0, 0.0, 0.19679058, 0.0],
                            [0.0, 0.0, 0.0, 0.0],
                            [0.62170676, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0],
                        ],
                        [
                            [0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.19679058, 0.0],
                            [0.0, 0.0, 0.0, 0.0],
                            [0.62170676, 0.0, 0.0, 0.0],
                        ],
                    ],
                    [
                        [
                            [0.0, 0.6264025, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.19679058],
                            [0.0, 0.0, 0.0, 0.0],
                        ],
                        [
                            [0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.6264025, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.19679058],
                        ],
                        [
                            [0.0, 0.0, 0.0, 0.19679058],
                            [0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.62170676, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0],
                        ],
                        [
                            [0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.19679058],
                            [0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.62170676, 0.0, 0.0],
                        ],
                    ],
                    [
                        [
                            [0.0, 0.0, 0.62170676, 0.0],
                            [0.0, 0.0, 0.0, 0.0],
                            [0.19679058, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0],
                        ],
                        [
                            [0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.62170676, 0.0],
                            [0.0, 0.0, 0.0, 0.0],
                            [0.19679058, 0.0, 0.0, 0.0],
                        ],
                        [
                            [0.19679058, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.65307075, 0.0],
                            [0.0, 0.0, 0.0, 0.0],
                        ],
                        [
                            [0.0, 0.0, 0.0, 0.0],
                            [0.19679058, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.65307075, 0.0],
                        ],
                    ],
                    [
                        [
                            [0.0, 0.0, 0.0, 0.62170676],
                            [0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.19679058, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0],
                        ],
                        [
                            [0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.62170676],
                            [0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.19679058, 0.0, 0.0],
                        ],
                        [
                            [0.0, 0.19679058, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.65307075],
                            [0.0, 0.0, 0.0, 0.0],
                        ],
                        [
                            [0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.19679058, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.65307075],
                        ],
                    ],
                ]
            )
        )
        mo_eint_set = SpinMOeIntSet(
            const=nuc_energy, mo_1e_int=spin_mo_1e_int, mo_2e_int=spin_mo_2e_int
        )

        jw_qubit_operator, _ = get_qubit_mapped_hamiltonian(cas(2, 2), mo_eint_set)

        expected_jw_qubit_operator = Operator(
            {
                PAULI_IDENTITY: -0.3276081896748091,
                pauli_label("Z0"): +0.13716572937099503,
                pauli_label("Z1"): +0.13716572937099503,
                pauli_label("Z2"): -0.1303629205710911,
                pauli_label("Z3"): -0.1303629205710911,
                pauli_label("Z0 Z1"): +0.15660062488237947,
                pauli_label("Z0 Z2"): +0.10622904490856078,
                pauli_label("Z0 Z3"): +0.15542669077992832,
                pauli_label("Z1 Z2"): +0.15542669077992832,
                pauli_label("Z1 Z3"): +0.10622904490856078,
                pauli_label("Z2 Z3"): +0.1632676867356435,
                pauli_label("X0 X1 Y2 Y3"): -0.04919764587136755,
                pauli_label("X0 Y1 Y2 X3"): +0.04919764587136755,
                pauli_label("Y0 X1 X2 Y3"): +0.04919764587136755,
                pauli_label("Y0 Y1 X2 X3"): -0.04919764587136755,
            }
        )

        assert truncate(expected_jw_qubit_operator - jw_qubit_operator) == zero()

        bk_qubit_operator, _ = get_qubit_mapped_hamiltonian(
            cas(2, 2), mo_eint_set, fermion_qubit_mapping=bravyi_kitaev
        )

        expected_bk_qubit_operator = Operator(
            {
                PAULI_IDENTITY: (-0.32760818967480876 + 0j),
                pauli_label("X0 Z1 X2"): (0.04919764587136755 + 0j),
                pauli_label("X0 Z1 X2 Z3"): (0.04919764587136755 + 0j),
                pauli_label("Y0 Z1 Y2"): (0.04919764587136755 + 0j),
                pauli_label("Y0 Z1 Y2 Z3"): (0.04919764587136755 + 0j),
                pauli_label("Z0"): (0.13716572937099514 + 0j),
                pauli_label("Z0 Z1"): (0.13716572937099503 + 0j),
                pauli_label("Z0 Z1 Z2"): (0.15542669077992832 + 0j),
                pauli_label("Z0 Z1 Z2 Z3"): (0.15542669077992832 + 0j),
                pauli_label("Z0 Z2"): (0.10622904490856078 + 0j),
                pauli_label("Z0 Z2 Z3"): (0.10622904490856078 + 0j),
                pauli_label("Z1"): (0.15660062488237947 + 0j),
                pauli_label("Z1 Z2 Z3"): (-0.1303629205710911 + 0j),
                pauli_label("Z1 Z3"): (0.1632676867356435 + 0j),
                pauli_label("Z2"): (-0.1303629205710911 + 0j),
            }
        )
        assert truncate(expected_bk_qubit_operator - bk_qubit_operator) == zero()

        scbk_qubit_operator, _ = get_qubit_mapped_hamiltonian(
            cas(2, 2),
            mo_eint_set,
            sz=0,
            fermion_qubit_mapping=symmetry_conserving_bravyi_kitaev,
        )
        expected_scbk_qubit_operator = Operator(
            {
                PAULI_IDENTITY: -0.5400662794919303,
                pauli_label("X0 X1"): 0.1967905834854702,
                pauli_label("Z0"): 0.26752864994208625,
                pauli_label("Z0 Z1"): 0.009014930058166337,
                pauli_label("Z1"): 0.26752864994208614,
            }
        )
        assert truncate(scbk_qubit_operator - expected_scbk_qubit_operator) == zero()

    def test_get_qubit_mapped_active_space_hamiltonian_operator(self) -> None:
        nuc_energy = -1.18702694476004
        spin_mo_1e_int = SpinMO1eIntArray(
            array([[-0.33696926, 0.0], [0.0, -0.33696926]])
        )
        spin_mo_2e_int = SpinMO2eIntArray(
            array(
                [
                    [[[0.53466412, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.53466412, 0.0]]],
                    [[[0.0, 0.53466412], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.53466412]]],
                ]
            )
        )
        mo_eint_set = SpinMOeIntSet(
            const=nuc_energy, mo_1e_int=spin_mo_1e_int, mo_2e_int=spin_mo_2e_int
        )

        jw_qubit_operator, _ = get_qubit_mapped_hamiltonian(cas(1, 1), mo_eint_set)
        expected_jw_qubit_operator = Operator(
            {
                PAULI_IDENTITY: -1.3903301703637687,
                pauli_label("Z0"): 0.03481859786069455,
                pauli_label("Z1"): 0.03481859786069455,
                pauli_label("Z0 Z1"): 0.13366602988233975,
            }
        )
        assert truncate(jw_qubit_operator - expected_jw_qubit_operator) == zero()

        bk_qubit_operator, _ = get_qubit_mapped_hamiltonian(
            cas(1, 1), mo_eint_set, fermion_qubit_mapping=bravyi_kitaev
        )
        expected_bk_qubit_operator = Operator(
            {
                PAULI_IDENTITY: -1.3903301747600398,
                pauli_label("Z0"): 0.03481859786069455,
                pauli_label("Z1"): 0.13366602988233975,
                pauli_label("Z0 Z1"): 0.03481859786069455,
            }
        )
        assert truncate(bk_qubit_operator - expected_bk_qubit_operator) == zero()
