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

from quri_parts.circuit import LinearMappedParametricQuantumCircuit
from quri_parts.openfermion.ansatz.kupccgsd import (
    KUpCCGSD,
    _generalized_pair_double_excitations,
    _generalized_single_excitations,
)
from quri_parts.openfermion.transforms import (
    jordan_wigner,
    symmetry_conserving_bravyi_kitaev,
)


def test_generalized_single_excitations() -> None:
    n_spin_orbitals = 4

    s_excs, s_names = _generalized_single_excitations(
        n_spin_orbitals, delta_sz=0, singlet_excitation=False, k=1
    )
    assert s_excs == [(0, 2), (1, 3)]
    assert s_names == [["t1_0_0_2", "t1_0_1_3"]]

    s_excs, s_names = _generalized_single_excitations(
        n_spin_orbitals, delta_sz=0, singlet_excitation=False, k=2
    )
    assert s_excs == [(0, 2), (1, 3)]
    assert s_names == [["t1_0_0_2", "t1_0_1_3"], ["t1_1_0_2", "t1_1_1_3"]]

    s_excs, s_names = _generalized_single_excitations(
        n_spin_orbitals, delta_sz=0, singlet_excitation=True, k=1
    )
    assert s_excs == [(0, 2), (1, 3)]
    assert s_names == [["spatialt1_0_0_1", "spatialt1_0_0_1"]]

    s_excs, s_names = _generalized_single_excitations(
        n_spin_orbitals, delta_sz=1, singlet_excitation=False, k=1
    )
    assert s_excs == [(1, 0), (1, 2), (3, 0), (3, 2)]
    assert s_names == [["t1_0_1_0", "t1_0_1_2", "t1_0_3_0", "t1_0_3_2"]]

    s_excs, s_names = _generalized_single_excitations(n_spin_orbitals, 1, False, 2)
    assert s_excs == [(1, 0), (1, 2), (3, 0), (3, 2)]
    assert s_names == [
        ["t1_0_1_0", "t1_0_1_2", "t1_0_3_0", "t1_0_3_2"],
        ["t1_1_1_0", "t1_1_1_2", "t1_1_3_0", "t1_1_3_2"],
    ]

    with pytest.raises(
        AssertionError, match="delta_sz can only be 0 when singlet_excitation is True."
    ):
        s_excs, s_names = _generalized_single_excitations(n_spin_orbitals, 1, True, 1)

    n_spin_orbitals = 6
    s_excs, s_names = _generalized_single_excitations(n_spin_orbitals, 0, False, 1)
    assert s_excs == [
        (0, 2),
        (0, 4),
        (1, 3),
        (1, 5),
        (2, 4),
        (3, 5),
    ]
    assert s_names == [
        ["t1_0_0_2", "t1_0_0_4", "t1_0_1_3", "t1_0_1_5", "t1_0_2_4", "t1_0_3_5"]
    ]

    s_excs, s_names = _generalized_single_excitations(
        n_spin_orbitals, delta_sz=0, singlet_excitation=False, k=2
    )
    assert s_excs == [
        (0, 2),
        (0, 4),
        (1, 3),
        (1, 5),
        (2, 4),
        (3, 5),
    ]
    assert s_names == [
        ["t1_0_0_2", "t1_0_0_4", "t1_0_1_3", "t1_0_1_5", "t1_0_2_4", "t1_0_3_5"],
        ["t1_1_0_2", "t1_1_0_4", "t1_1_1_3", "t1_1_1_5", "t1_1_2_4", "t1_1_3_5"],
    ]

    s_excs, s_names = _generalized_single_excitations(
        n_spin_orbitals, delta_sz=-1, singlet_excitation=False, k=1
    )
    assert s_excs == [
        (0, 1),
        (0, 3),
        (0, 5),
        (2, 1),
        (2, 3),
        (2, 5),
        (4, 1),
        (4, 3),
        (4, 5),
    ]
    assert s_names == [
        [
            "t1_0_0_1",
            "t1_0_0_3",
            "t1_0_0_5",
            "t1_0_2_1",
            "t1_0_2_3",
            "t1_0_2_5",
            "t1_0_4_1",
            "t1_0_4_3",
            "t1_0_4_5",
        ]
    ]

    s_excs, s_names = _generalized_single_excitations(
        n_spin_orbitals, delta_sz=-1, singlet_excitation=False, k=2
    )
    assert s_excs == [
        (0, 1),
        (0, 3),
        (0, 5),
        (2, 1),
        (2, 3),
        (2, 5),
        (4, 1),
        (4, 3),
        (4, 5),
    ]
    assert s_names == [
        [
            "t1_0_0_1",
            "t1_0_0_3",
            "t1_0_0_5",
            "t1_0_2_1",
            "t1_0_2_3",
            "t1_0_2_5",
            "t1_0_4_1",
            "t1_0_4_3",
            "t1_0_4_5",
        ],
        [
            "t1_1_0_1",
            "t1_1_0_3",
            "t1_1_0_5",
            "t1_1_2_1",
            "t1_1_2_3",
            "t1_1_2_5",
            "t1_1_4_1",
            "t1_1_4_3",
            "t1_1_4_5",
        ],
    ]


def test_generalized_pair_double_excitations() -> None:
    n_spin_orbitals = 4
    d_excs, d_names = _generalized_pair_double_excitations(n_spin_orbitals, k=1)
    assert d_excs == [(0, 1, 2, 3)]
    assert d_names == [["t2_0_0_1_2_3"]]

    d_excs, d_names = _generalized_pair_double_excitations(n_spin_orbitals, k=2)
    assert d_excs == [(0, 1, 2, 3)]
    assert d_names == [["t2_0_0_1_2_3"], ["t2_1_0_1_2_3"]]

    n_spin_orbitals = 6
    d_excs, d_names = _generalized_pair_double_excitations(n_spin_orbitals, k=1)
    assert d_excs == [
        (0, 1, 2, 3),
        (0, 1, 4, 5),
        (2, 3, 4, 5),
    ]
    assert d_names == [["t2_0_0_1_2_3", "t2_0_0_1_4_5", "t2_0_2_3_4_5"]]

    d_excs, d_names = _generalized_pair_double_excitations(n_spin_orbitals, k=2)
    assert d_excs == [
        (0, 1, 2, 3),
        (0, 1, 4, 5),
        (2, 3, 4, 5),
    ]
    assert d_names == [
        ["t2_0_0_1_2_3", "t2_0_0_1_4_5", "t2_0_2_3_4_5"],
        ["t2_1_0_1_2_3", "t2_1_0_1_4_5", "t2_1_2_3_4_5"],
    ]


class TestKUpCCGSD:
    def test_invalid_kupccgsd(self) -> None:
        with pytest.raises(AssertionError):
            KUpCCGSD(4, fermion_qubit_mapping=jordan_wigner(8))

    def test_kupccgsd_k1_trotter1(self) -> None:
        n_spin_orbitals = 4
        ansatz = KUpCCGSD(n_spin_orbitals)
        expected_ansatz = LinearMappedParametricQuantumCircuit(n_spin_orbitals)
        params = expected_ansatz.add_parameters(*[f"param{i}" for i in range(3)])
        expected_ansatz.add_ParametricPauliRotation_gate(
            (0, 1, 2), (2, 3, 1), {params[0]: -1}
        )
        expected_ansatz.add_ParametricPauliRotation_gate(
            (0, 1, 2), (1, 3, 2), {params[0]: +1}
        )
        expected_ansatz.add_ParametricPauliRotation_gate(
            (2, 1, 3), (3, 2, 1), {params[1]: -1}
        )
        expected_ansatz.add_ParametricPauliRotation_gate(
            (2, 1, 3), (3, 1, 2), {params[1]: +1}
        )
        expected_ansatz.add_ParametricPauliRotation_gate(
            (0, 1, 3, 2), (1, 1, 1, 2), {params[2]: -0.25}
        )
        expected_ansatz.add_ParametricPauliRotation_gate(
            (3, 0, 1, 2), (1, 2, 2, 2), {params[2]: +0.25}
        )
        expected_ansatz.add_ParametricPauliRotation_gate(
            (1, 3, 0, 2), (1, 1, 2, 1), {params[2]: +0.25}
        )
        expected_ansatz.add_ParametricPauliRotation_gate(
            (0, 3, 1, 2), (1, 1, 2, 1), {params[2]: +0.25}
        )
        expected_ansatz.add_ParametricPauliRotation_gate(
            (1, 0, 3, 2), (1, 2, 2, 2), {params[2]: -0.25}
        )
        expected_ansatz.add_ParametricPauliRotation_gate(
            (0, 3, 1, 2), (1, 2, 2, 2), {params[2]: -0.25}
        )
        expected_ansatz.add_ParametricPauliRotation_gate(
            (0, 1, 2, 3), (1, 1, 1, 2), {params[2]: -0.25}
        )
        expected_ansatz.add_ParametricPauliRotation_gate(
            (0, 1, 2, 3), (2, 2, 1, 2), {params[2]: +0.25}
        )

        assert ansatz.parameter_count == expected_ansatz.parameter_count
        assert ansatz._circuit.gates == expected_ansatz._circuit.gates
        param_vals = [0.1 * (i + 1) for i in range(ansatz.parameter_count)]
        bound_ansatz = ansatz.bind_parameters(param_vals)
        expected_bound_ansatz = expected_ansatz.bind_parameters(param_vals)
        assert bound_ansatz == expected_bound_ansatz

    def test_kupccgsd_k2_trotter2_scbk(self) -> None:
        n_spin_orbitals = 4
        n_electrons = 2
        operator_mapping = symmetry_conserving_bravyi_kitaev(
            n_spin_orbitals, n_electrons, 0.0
        )
        k = 2
        ansatz = KUpCCGSD(
            n_spin_orbitals,
            k,
            operator_mapping,
            trotter_number=2,
        )
        expected_ansatz = LinearMappedParametricQuantumCircuit(n_spin_orbitals - 2)
        params = expected_ansatz.add_parameters(*[f"param{i}" for i in range(6)])
        for i in range(k):
            expected_ansatz.add_ParametricPauliRotation_gate(
                qubit_indices=(0,),
                pauli_ids=(2,),
                angle={params[i * 3 + 0]: -1},
            )
            expected_ansatz.add_ParametricPauliRotation_gate(
                qubit_indices=(1,),
                pauli_ids=(2,),
                angle={params[i * 3 + 1]: -1},
            )
            expected_ansatz.add_ParametricPauliRotation_gate(
                qubit_indices=(0, 1),
                pauli_ids=(1, 2),
                angle={params[i * 3 + 2]: 0.5},
            )
            expected_ansatz.add_ParametricPauliRotation_gate(
                qubit_indices=(1, 0),
                pauli_ids=(1, 2),
                angle={params[i * 3 + 2]: 0.5},
            )
            expected_ansatz.add_ParametricPauliRotation_gate(
                qubit_indices=(0,),
                pauli_ids=(2,),
                angle={params[i * 3 + 0]: -1},
            )
            expected_ansatz.add_ParametricPauliRotation_gate(
                qubit_indices=(1,),
                pauli_ids=(2,),
                angle={params[i * 3 + 1]: -1},
            )
            expected_ansatz.add_ParametricPauliRotation_gate(
                qubit_indices=(0, 1),
                pauli_ids=(1, 2),
                angle={params[i * 3 + 2]: 0.5},
            )
            expected_ansatz.add_ParametricPauliRotation_gate(
                qubit_indices=(1, 0),
                pauli_ids=(1, 2),
                angle={params[i * 3 + 2]: 0.5},
            )
        assert ansatz.parameter_count == expected_ansatz.parameter_count
        assert ansatz._circuit.gates == expected_ansatz._circuit.gates
        param_vals = [0.1 * (i + 1) for i in range(ansatz.parameter_count)]
        bound_ansatz = ansatz.bind_parameters(param_vals)
        expected_bound_ansatz = expected_ansatz.bind_parameters(param_vals)
        assert bound_ansatz == expected_bound_ansatz


class TestSingletExcitedKUpCCGSD:
    def test_singlet_excited_kupccgsd_k1_trotter1(self) -> None:
        n_spin_orbitals = 4
        ansatz = KUpCCGSD(n_spin_orbitals, singlet_excitation=True)
        expected_ansatz = LinearMappedParametricQuantumCircuit(n_spin_orbitals)
        params = expected_ansatz.add_parameters(*[f"param{i}" for i in range(2)])
        expected_ansatz.add_ParametricPauliRotation_gate(
            (0, 1, 2), (2, 3, 1), {params[0]: -1}
        )
        expected_ansatz.add_ParametricPauliRotation_gate(
            (0, 1, 2), (1, 3, 2), {params[0]: +1}
        )
        expected_ansatz.add_ParametricPauliRotation_gate(
            (2, 1, 3), (3, 2, 1), {params[0]: -1}
        )
        expected_ansatz.add_ParametricPauliRotation_gate(
            (2, 1, 3), (3, 1, 2), {params[0]: +1}
        )
        expected_ansatz.add_ParametricPauliRotation_gate(
            (0, 1, 3, 2), (1, 1, 1, 2), {params[1]: -0.25}
        )
        expected_ansatz.add_ParametricPauliRotation_gate(
            (3, 0, 1, 2), (1, 2, 2, 2), {params[1]: +0.25}
        )
        expected_ansatz.add_ParametricPauliRotation_gate(
            (1, 3, 0, 2), (1, 1, 2, 1), {params[1]: +0.25}
        )
        expected_ansatz.add_ParametricPauliRotation_gate(
            (0, 3, 1, 2), (1, 1, 2, 1), {params[1]: +0.25}
        )
        expected_ansatz.add_ParametricPauliRotation_gate(
            (1, 0, 3, 2), (1, 2, 2, 2), {params[1]: -0.25}
        )
        expected_ansatz.add_ParametricPauliRotation_gate(
            (0, 3, 1, 2), (1, 2, 2, 2), {params[1]: -0.25}
        )
        expected_ansatz.add_ParametricPauliRotation_gate(
            (0, 1, 2, 3), (1, 1, 1, 2), {params[1]: -0.25}
        )
        expected_ansatz.add_ParametricPauliRotation_gate(
            (0, 1, 2, 3), (2, 2, 1, 2), {params[1]: +0.25}
        )

        assert ansatz.parameter_count == expected_ansatz.parameter_count
        assert ansatz._circuit.gates == expected_ansatz._circuit.gates
        param_vals = [0.1 * (i + 1) for i in range(ansatz.parameter_count)]
        bound_ansatz = ansatz.bind_parameters(param_vals)
        expected_bound_ansatz = expected_ansatz.bind_parameters(param_vals)
        assert bound_ansatz == expected_bound_ansatz

        param_name_set = set()
        for param in ansatz.param_mapping.in_params:
            param_name_set.add(param.name)
        assert param_name_set == {"spatialt1_0_0_1", "t2_0_0_1_2_3"}

    def test_singlet_excited_kupccgsd_k2_trotter2_scbk(self) -> None:
        n_spin_orbitals = 4
        n_electrons = 2
        fermion_qubit_mapping = symmetry_conserving_bravyi_kitaev(
            n_spin_orbitals, n_electrons, 0.0
        )
        k = 2
        ansatz = KUpCCGSD(
            n_spin_orbitals,
            k,
            fermion_qubit_mapping,
            trotter_number=2,
            singlet_excitation=True,
        )
        expected_ansatz = LinearMappedParametricQuantumCircuit(n_spin_orbitals - 2)
        params = expected_ansatz.add_parameters(*[f"param{i}" for i in range(4)])
        for i in range(k):
            expected_ansatz.add_ParametricPauliRotation_gate(
                qubit_indices=(0,),
                pauli_ids=(2,),
                angle={params[i * 2 + 0]: -1},
            )
            expected_ansatz.add_ParametricPauliRotation_gate(
                qubit_indices=(1,),
                pauli_ids=(2,),
                angle={params[i * 2 + 0]: -1},
            )
            expected_ansatz.add_ParametricPauliRotation_gate(
                qubit_indices=(0, 1),
                pauli_ids=(1, 2),
                angle={params[i * 2 + 1]: 0.5},
            )
            expected_ansatz.add_ParametricPauliRotation_gate(
                qubit_indices=(1, 0),
                pauli_ids=(1, 2),
                angle={params[i * 2 + 1]: 0.5},
            )
            expected_ansatz.add_ParametricPauliRotation_gate(
                qubit_indices=(0,),
                pauli_ids=(2,),
                angle={params[i * 2 + 0]: -1},
            )
            expected_ansatz.add_ParametricPauliRotation_gate(
                qubit_indices=(1,),
                pauli_ids=(2,),
                angle={params[i * 2 + 0]: -1},
            )
            expected_ansatz.add_ParametricPauliRotation_gate(
                qubit_indices=(0, 1),
                pauli_ids=(1, 2),
                angle={params[i * 2 + 1]: 0.5},
            )
            expected_ansatz.add_ParametricPauliRotation_gate(
                qubit_indices=(1, 0),
                pauli_ids=(1, 2),
                angle={params[i * 2 + 1]: 0.5},
            )
        assert ansatz.parameter_count == expected_ansatz.parameter_count
        assert ansatz._circuit.gates == expected_ansatz._circuit.gates
        param_vals = [0.1 * (i + 1) for i in range(ansatz.parameter_count)]
        bound_ansatz = ansatz.bind_parameters(param_vals)
        expected_bound_ansatz = expected_ansatz.bind_parameters(param_vals)
        assert bound_ansatz == expected_bound_ansatz

        param_name_set = set()
        for param in ansatz.param_mapping.in_params:
            param_name_set.add(param.name)
        assert param_name_set == {
            "spatialt1_0_0_1",
            "t2_0_0_1_2_3",
            "spatialt1_1_0_1",
            "t2_1_0_1_2_3",
        }
