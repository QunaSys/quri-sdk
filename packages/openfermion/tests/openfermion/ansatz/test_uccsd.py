# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Sequence

import pytest
from openfermion import FermionOperator

from quri_parts.chem.utils.excitations import excitations
from quri_parts.circuit import LinearMappedParametricQuantumCircuit
from quri_parts.core.circuit import add_parametric_commuting_paulis_exp_gate
from quri_parts.core.operator import Operator, pauli_label, truncate
from quri_parts.openfermion.ansatz.uccsd import (
    TrotterUCCSD,
    _construct_circuit,
    _construct_singlet_excitation_circuit,
)
from quri_parts.openfermion.transforms import (
    OpenFermionQubitMapping,
    bravyi_kitaev,
    jordan_wigner,
    symmetry_conserving_bravyi_kitaev,
)
from quri_parts.openfermion.utils import add_exp_excitation_gates_trotter_decomposition
from quri_parts.openfermion.utils.add_exp_excitation_gates_trotter_decomposition import (  # noqa: E501
    create_anti_hermitian_sd_excitation_operator,
)


class TestConstructCircuit:
    def test_construct_circuit_w_singles_trotter1(self) -> None:
        n_spin_orbitals = 4
        n_electrons = 2
        fermion_qubit_mapping = jordan_wigner(n_spin_orbitals, n_electrons)
        trotter_number = 1
        use_singles = True
        circuit = _construct_circuit(
            fermion_qubit_mapping,
            trotter_number,
            use_singles,
        )
        expected_circuit = LinearMappedParametricQuantumCircuit(n_spin_orbitals)
        params = expected_circuit.add_parameters("param1", "param2", "param3")
        op_mapper = fermion_qubit_mapping.of_operator_mapper
        s_excs, d_excs = excitations(n_spin_orbitals, n_electrons)
        add_exp_excitation_gates_trotter_decomposition(
            expected_circuit, d_excs, [params[-1]], op_mapper, 1 / trotter_number
        )
        add_exp_excitation_gates_trotter_decomposition(
            expected_circuit, s_excs, params[:-1], op_mapper, 1 / trotter_number
        )
        assert circuit.parameter_count == expected_circuit.parameter_count
        assert circuit._circuit.gates == expected_circuit._circuit.gates
        param_vals = [0.1 * (i + 1) for i in range(circuit.parameter_count)]
        bound_circuit = circuit.bind_parameters(param_vals)
        expected_bound_circuit = expected_circuit.bind_parameters(param_vals)
        assert bound_circuit == expected_bound_circuit

    def test_construct_circuit_wo_singles_trotter1(self) -> None:
        n_spin_orbitals = 4
        n_electrons = 2
        fermion_qubit_mapping = jordan_wigner(n_spin_orbitals, n_electrons)
        trotter_number = 1
        use_singles = False

        circuit = _construct_circuit(
            fermion_qubit_mapping,
            trotter_number,
            use_singles,
        )
        expected_circuit = LinearMappedParametricQuantumCircuit(n_spin_orbitals)
        param = expected_circuit.add_parameter("param")
        op_mapper = fermion_qubit_mapping.of_operator_mapper
        _, d_excs = excitations(n_spin_orbitals, n_electrons)
        add_exp_excitation_gates_trotter_decomposition(
            expected_circuit, d_excs, [param], op_mapper, 1 / trotter_number
        )
        assert circuit.parameter_count == expected_circuit.parameter_count
        assert circuit._circuit.gates == expected_circuit._circuit.gates
        param_vals = [0.1 * (i + 1) for i in range(circuit.parameter_count)]
        bound_circuit = circuit.bind_parameters(param_vals)
        expected_bound_circuit = expected_circuit.bind_parameters(param_vals)
        assert bound_circuit == expected_bound_circuit

    def test_construct_circuit_w_singles_trotter2_scbk(self) -> None:
        n_spin_orbitals = 4
        n_electrons = 2
        fermion_qubit_mapping = symmetry_conserving_bravyi_kitaev(
            n_spin_orbitals, n_electrons, 0.0
        )
        use_singles = True
        trotter_number = 2

        circuit = _construct_circuit(
            fermion_qubit_mapping,
            trotter_number,
            use_singles,
        )
        n_qubits = fermion_qubit_mapping.n_qubits
        assert isinstance(n_qubits, int)
        op_mapper = fermion_qubit_mapping.of_operator_mapper
        expected_circuit = LinearMappedParametricQuantumCircuit(n_qubits)
        params = expected_circuit.add_parameters("param1", "param2", "param3")
        s_excs, d_excs = excitations(n_spin_orbitals, n_electrons)
        add_exp_excitation_gates_trotter_decomposition(
            expected_circuit, d_excs, [params[-1]], op_mapper, 1 / trotter_number
        )
        add_exp_excitation_gates_trotter_decomposition(
            expected_circuit, s_excs, params[:-1], op_mapper, 1 / trotter_number
        )
        add_exp_excitation_gates_trotter_decomposition(
            expected_circuit, d_excs, [params[-1]], op_mapper, 1 / trotter_number
        )
        add_exp_excitation_gates_trotter_decomposition(
            expected_circuit, s_excs, params[:-1], op_mapper, 1 / trotter_number
        )
        assert circuit.parameter_count == expected_circuit.parameter_count
        assert circuit._circuit.gates == expected_circuit._circuit.gates
        param_vals = [0.1 * (i + 1) for i in range(circuit.parameter_count)]
        bound_circuit = circuit.bind_parameters(param_vals)
        expected_bound_circuit = expected_circuit.bind_parameters(param_vals)
        assert bound_circuit == expected_bound_circuit


class TestConstructSpinSymmetricCircuit:
    def test_construct_circuit_w_singles_trotter1(self) -> None:
        n_spin_orbitals = 8
        n_electrons = 4
        fermion_qubit_mapping = jordan_wigner(n_spin_orbitals, n_electrons)
        trotter_number = 1
        use_singles = True

        circuit = _construct_singlet_excitation_circuit(
            fermion_qubit_mapping,
            trotter_number,
            use_singles,
        )
        expected_circuit = LinearMappedParametricQuantumCircuit(n_spin_orbitals)
        s_0_2 = expected_circuit.add_parameter("s_0_2")
        s_0_3 = expected_circuit.add_parameter("s_0_3")
        s_1_2 = expected_circuit.add_parameter("s_1_2")
        s_1_3 = expected_circuit.add_parameter("s_1_3")
        d_0_0_2_2 = expected_circuit.add_parameter("d_0_0_2_2")
        d_0_0_2_3 = expected_circuit.add_parameter("d_0_0_2_3")
        d_0_0_3_3 = expected_circuit.add_parameter("d_0_0_3_3")
        d_0_1_2_2 = expected_circuit.add_parameter("d_0_1_2_2")
        d_0_1_2_3 = expected_circuit.add_parameter("d_0_1_2_3")
        d_0_1_3_2 = expected_circuit.add_parameter("d_0_1_3_2")
        d_0_1_3_3 = expected_circuit.add_parameter("d_0_1_3_3")
        d_1_1_2_2 = expected_circuit.add_parameter("d_1_1_2_2")
        d_1_1_2_3 = expected_circuit.add_parameter("d_1_1_2_3")
        d_1_1_3_3 = expected_circuit.add_parameter("d_1_1_3_3")

        op_mapper = fermion_qubit_mapping.of_operator_mapper

        operator_0_4 = (
            create_anti_hermitian_sd_excitation_operator((0, 4), op_mapper) * -1j
        )
        add_parametric_commuting_paulis_exp_gate(
            expected_circuit, {s_0_2: 1}, operator_0_4, 1
        )
        operator_0_6 = (
            create_anti_hermitian_sd_excitation_operator((0, 6), op_mapper) * -1j
        )
        add_parametric_commuting_paulis_exp_gate(
            expected_circuit, {s_0_3: 1}, operator_0_6, 1
        )
        operator_1_5 = (
            create_anti_hermitian_sd_excitation_operator((1, 5), op_mapper) * -1j
        )
        add_parametric_commuting_paulis_exp_gate(
            expected_circuit, {s_0_2: 1}, operator_1_5, 1
        )
        operator_1_7 = (
            create_anti_hermitian_sd_excitation_operator((1, 7), op_mapper) * -1j
        )
        add_parametric_commuting_paulis_exp_gate(
            expected_circuit, {s_0_3: 1}, operator_1_7, 1
        )
        operator_2_4 = (
            create_anti_hermitian_sd_excitation_operator((2, 4), op_mapper) * -1j
        )
        add_parametric_commuting_paulis_exp_gate(
            expected_circuit, {s_1_2: 1}, operator_2_4, 1
        )
        operator_2_6 = (
            create_anti_hermitian_sd_excitation_operator((2, 6), op_mapper) * -1j
        )
        add_parametric_commuting_paulis_exp_gate(
            expected_circuit, {s_1_3: 1}, operator_2_6, 1
        )
        operator_3_5 = (
            create_anti_hermitian_sd_excitation_operator((3, 5), op_mapper) * -1j
        )
        add_parametric_commuting_paulis_exp_gate(
            expected_circuit, {s_1_2: 1}, operator_3_5, 1
        )
        operator_3_7 = (
            create_anti_hermitian_sd_excitation_operator((3, 7), op_mapper) * -1j
        )
        add_parametric_commuting_paulis_exp_gate(
            expected_circuit, {s_1_3: 1}, operator_3_7, 1
        )

        operator_0_1_5_4 = (
            create_anti_hermitian_sd_excitation_operator((0, 1, 5, 4), op_mapper) * -1j
        )
        add_parametric_commuting_paulis_exp_gate(
            expected_circuit, {d_0_0_2_2: 1}, operator_0_1_5_4, 1
        )
        operator_0_1_7_4 = (
            create_anti_hermitian_sd_excitation_operator((0, 1, 7, 4), op_mapper) * -1j
        )
        add_parametric_commuting_paulis_exp_gate(
            expected_circuit, {d_0_0_2_3: 1}, operator_0_1_7_4, 1
        )
        operator_0_1_5_6 = (
            create_anti_hermitian_sd_excitation_operator((0, 1, 5, 6), op_mapper) * -1j
        )
        add_parametric_commuting_paulis_exp_gate(
            expected_circuit, {d_0_0_2_3: 1}, operator_0_1_5_6, 1
        )
        operator_0_1_7_6 = (
            create_anti_hermitian_sd_excitation_operator((0, 1, 7, 6), op_mapper) * -1j
        )
        add_parametric_commuting_paulis_exp_gate(
            expected_circuit, {d_0_0_3_3: 1}, operator_0_1_7_6, 1
        )
        operator_0_3_5_4 = (
            create_anti_hermitian_sd_excitation_operator((0, 3, 5, 4), op_mapper) * -1j
        )
        add_parametric_commuting_paulis_exp_gate(
            expected_circuit, {d_0_1_2_2: 1}, operator_0_3_5_4, 1
        )
        operator_0_3_7_4 = (
            create_anti_hermitian_sd_excitation_operator((0, 3, 7, 4), op_mapper) * -1j
        )
        add_parametric_commuting_paulis_exp_gate(
            expected_circuit, {d_0_1_2_3: 1}, operator_0_3_7_4, 1
        )
        operator_0_3_5_6 = (
            create_anti_hermitian_sd_excitation_operator((0, 3, 5, 6), op_mapper) * -1j
        )
        add_parametric_commuting_paulis_exp_gate(
            expected_circuit, {d_0_1_3_2: 1}, operator_0_3_5_6, 1
        )
        operator_0_3_7_6 = (
            create_anti_hermitian_sd_excitation_operator((0, 3, 7, 6), op_mapper) * -1j
        )
        add_parametric_commuting_paulis_exp_gate(
            expected_circuit, {d_0_1_3_3: 1}, operator_0_3_7_6, 1
        )
        operator_2_1_5_4 = (
            create_anti_hermitian_sd_excitation_operator((2, 1, 5, 4), op_mapper) * -1j
        )
        add_parametric_commuting_paulis_exp_gate(
            expected_circuit, {d_0_1_2_2: 1}, operator_2_1_5_4, 1
        )
        operator_2_1_7_4 = (
            create_anti_hermitian_sd_excitation_operator((2, 1, 7, 4), op_mapper) * -1j
        )
        add_parametric_commuting_paulis_exp_gate(
            expected_circuit, {d_0_1_3_2: 1}, operator_2_1_7_4, 1
        )
        operator_2_1_5_6 = (
            create_anti_hermitian_sd_excitation_operator((2, 1, 5, 6), op_mapper) * -1j
        )
        add_parametric_commuting_paulis_exp_gate(
            expected_circuit, {d_0_1_2_3: 1}, operator_2_1_5_6, 1
        )
        operator_2_1_7_6 = (
            create_anti_hermitian_sd_excitation_operator((2, 1, 7, 6), op_mapper) * -1j
        )
        add_parametric_commuting_paulis_exp_gate(
            expected_circuit, {d_0_1_3_3: 1}, operator_2_1_7_6, 1
        )
        operator_2_3_5_4 = (
            create_anti_hermitian_sd_excitation_operator((2, 3, 5, 4), op_mapper) * -1j
        )
        add_parametric_commuting_paulis_exp_gate(
            expected_circuit, {d_1_1_2_2: 1}, operator_2_3_5_4, 1
        )
        operator_2_3_7_4 = (
            create_anti_hermitian_sd_excitation_operator((2, 3, 7, 4), op_mapper) * -1j
        )
        add_parametric_commuting_paulis_exp_gate(
            expected_circuit, {d_1_1_2_3: 1}, operator_2_3_7_4, 1
        )
        operator_2_3_5_6 = (
            create_anti_hermitian_sd_excitation_operator((2, 3, 5, 6), op_mapper) * -1j
        )
        add_parametric_commuting_paulis_exp_gate(
            expected_circuit, {d_1_1_2_3: 1}, operator_2_3_5_6, 1
        )
        operator_2_3_7_6 = (
            create_anti_hermitian_sd_excitation_operator((2, 3, 7, 6), op_mapper) * -1j
        )
        add_parametric_commuting_paulis_exp_gate(
            expected_circuit, {d_1_1_3_3: 1}, operator_2_3_7_6, 1
        )
        operator_0_2_6_4 = (
            create_anti_hermitian_sd_excitation_operator((0, 2, 6, 4), op_mapper) * -1j
        )
        add_parametric_commuting_paulis_exp_gate(
            expected_circuit, {d_0_1_2_3: 1, d_0_1_3_2: -1}, operator_0_2_6_4, 1
        )
        operator_1_3_7_5 = (
            create_anti_hermitian_sd_excitation_operator((1, 3, 7, 5), op_mapper) * -1j
        )
        add_parametric_commuting_paulis_exp_gate(
            expected_circuit, {d_0_1_2_3: 1, d_0_1_3_2: -1}, operator_1_3_7_5, 1
        )

        assert circuit.parameter_count == expected_circuit.parameter_count == 14
        assert circuit._circuit.gates == expected_circuit._circuit.gates
        param_vals = [0.1 * (i + 1) for i in range(circuit.parameter_count)]
        bound_circuit = circuit.bind_parameters(param_vals)
        expected_bound_circuit = expected_circuit.bind_parameters(param_vals)
        assert bound_circuit.gates == expected_bound_circuit.gates


class TestUCCSD:
    def test_trotter_singlet_uccsd_w_singles_jw(self) -> None:
        n_spin_orbitals = 4
        n_electrons = 2
        fermion_qubit_mapping = jordan_wigner(n_spin_orbitals, n_electrons)
        trotter_number = 1
        ansatz = TrotterUCCSD(
            n_spin_orbitals,
            n_electrons,
            fermion_qubit_mapping=fermion_qubit_mapping,
            trotter_number=trotter_number,
        )
        expected_ansatz = _construct_circuit(
            jordan_wigner(n_spin_orbitals, n_electrons),
            trotter_number=trotter_number,
            use_singles=True,
        )
        assert ansatz.parameter_count == expected_ansatz.parameter_count
        assert ansatz._circuit.gates == expected_ansatz._circuit.gates
        param_vals = [0.1 * (i + 1) for i in range(ansatz.parameter_count)]
        bound_ansatz = ansatz.bind_parameters(param_vals)
        expected_bound_ansatz = expected_ansatz.bind_parameters(param_vals)
        assert bound_ansatz == expected_bound_ansatz

    def test_trotter_singlet_uccsd_wo_singles_bk(self) -> None:
        n_spin_orbitals = 4
        n_electrons = 2
        trotter_number = 1
        ansatz = TrotterUCCSD(
            n_spin_orbitals,
            n_electrons,
            fermion_qubit_mapping=bravyi_kitaev(n_spin_orbitals, n_electrons),
            trotter_number=trotter_number,
            use_singles=False,
        )
        expected_ansatz = _construct_circuit(
            bravyi_kitaev(n_spin_orbitals, n_electrons),
            trotter_number=trotter_number,
            use_singles=False,
        )
        assert ansatz.parameter_count == expected_ansatz.parameter_count
        assert ansatz._circuit.gates == expected_ansatz._circuit.gates
        param_vals = [0.1 * (i + 1) for i in range(ansatz.parameter_count)]
        bound_ansatz = ansatz.bind_parameters(param_vals)
        expected_bound_ansatz = expected_ansatz.bind_parameters(param_vals)
        assert bound_ansatz == expected_bound_ansatz

    def test_trotter_singlet_uccsd_scbk_trotter2(self) -> None:
        n_spin_orbitals = 4
        n_electrons = 2
        trotter_number = 2
        ansatz = TrotterUCCSD(
            n_spin_orbitals,
            n_electrons,
            fermion_qubit_mapping=symmetry_conserving_bravyi_kitaev(
                n_spin_orbitals, n_electrons, 0.0
            ),
            trotter_number=trotter_number,
        )
        expected_ansatz = _construct_circuit(
            symmetry_conserving_bravyi_kitaev(n_spin_orbitals, n_electrons, 0.0),
            trotter_number=trotter_number,
            use_singles=True,
        )
        assert ansatz.parameter_count == expected_ansatz.parameter_count
        assert ansatz._circuit.gates == expected_ansatz._circuit.gates
        param_vals = [0.1 * (i + 1) for i in range(ansatz.parameter_count)]
        bound_ansatz = ansatz.bind_parameters(param_vals)
        expected_bound_ansatz = expected_ansatz.bind_parameters(param_vals)
        assert bound_ansatz == expected_bound_ansatz

    def test_singlet_uccsd_invalid_input(self) -> None:
        with pytest.raises(
            ValueError,
            match=(
                "Singlet excitation is not supported when " "number of electron is odd."
            ),
        ):
            TrotterUCCSD(4, 3, singlet_excitation=True)
        with pytest.raises(ValueError):
            TrotterUCCSD(4, 4)
        with pytest.raises(AssertionError):
            TrotterUCCSD(4, 4, fermion_qubit_mapping=jordan_wigner(8))
        with pytest.raises(AssertionError):
            TrotterUCCSD(4, 4, fermion_qubit_mapping=jordan_wigner(4, 2))


class TestSingletUCCSD:
    @staticmethod
    def check_is_singlet(
        uccsd_ansatz: TrotterUCCSD,
        parameters: Sequence[float],
        operator_mapping: OpenFermionQubitMapping,
    ) -> bool:
        # Build spin operators
        Sx = 0
        for i in range(0, 8, 2):
            Sx += 0.5 * FermionOperator([(i + 1, 1)]) * FermionOperator([(i, 0)])
            Sx += 0.5 * FermionOperator([(i, 1)]) * FermionOperator([(i + 1, 0)])

        Sy = 0
        for i in range(0, 8, 2):
            Sy += -0.5j * FermionOperator([(i, 1)]) * FermionOperator([(i + 1, 0)])
            Sy += 0.5j * FermionOperator([(i + 1, 1)]) * FermionOperator([(i, 0)])

        Sz = 0
        for i in range(8):
            Sz += (
                (-1) ** (i % 2) * FermionOperator([(i, 1)]) * FermionOperator([(i, 0)])
            )

        operator_mapper = operator_mapping.of_operator_mapper
        s2_operator = operator_mapper(Sx * Sx + Sy * Sy + Sz * Sz)

        # Sum all exponents together
        pauli_map = {1: "X", 2: "Y", 3: "Z"}
        bound_ansatz = uccsd_ansatz.bind_parameters(parameters)

        uccsd_excitation_generator = Operator({})
        for g in bound_ansatz.gates:
            pauli_str = ""
            for idx, pauli in zip(g.target_indices, g.pauli_ids):
                pauli_str = pauli_str + (pauli_map[pauli] + str(idx)) + " "
            uccsd_excitation_generator.add_term(pauli_label(pauli_str), g.params[0])

        commutator = (
            uccsd_excitation_generator * s2_operator
            - s2_operator * uccsd_excitation_generator
        )
        commutator = truncate(commutator, atol=1e-15)
        return len(commutator) == 0

    def test_sinlget_excited_uccsd_trotter_1(self) -> None:
        n_spin_orbitals = 8
        n_electrons = 4
        fermion_qubit_mapping = jordan_wigner(n_spin_orbitals, n_electrons)
        trotter_number = 1
        ansatz = TrotterUCCSD(
            n_spin_orbitals,
            n_electrons,
            fermion_qubit_mapping=fermion_qubit_mapping,
            trotter_number=trotter_number,
            singlet_excitation=True,
        )
        expected_ansatz = _construct_singlet_excitation_circuit(
            fermion_qubit_mapping,
            trotter_number=trotter_number,
            use_singles=True,
        )
        assert ansatz.parameter_count == expected_ansatz.parameter_count
        assert ansatz._circuit.gates == expected_ansatz._circuit.gates
        param_vals = [0.1 * (i + 1) for i in range(ansatz.parameter_count)]
        bound_ansatz = ansatz.bind_parameters(param_vals)
        expected_bound_ansatz = expected_ansatz.bind_parameters(param_vals)
        assert bound_ansatz == expected_bound_ansatz

        assert self.check_is_singlet(
            ansatz,
            param_vals,
            operator_mapping=jordan_wigner(n_spin_orbitals, n_electrons),
        )

    def test_sinlget_excited_uccsd_trotter_2(self) -> None:
        n_spin_orbitals = 8
        n_electrons = 4
        trotter_number = 1
        ansatz = TrotterUCCSD(
            n_spin_orbitals,
            n_electrons,
            fermion_qubit_mapping=bravyi_kitaev(n_spin_orbitals, n_electrons),
            trotter_number=trotter_number,
            singlet_excitation=True,
        )
        expected_ansatz = _construct_singlet_excitation_circuit(
            bravyi_kitaev(n_spin_orbitals, n_electrons),
            trotter_number=trotter_number,
            use_singles=True,
        )
        assert ansatz.parameter_count == expected_ansatz.parameter_count
        assert ansatz._circuit.gates == expected_ansatz._circuit.gates
        param_vals = [0.1 * (i + 1) for i in range(ansatz.parameter_count)]
        bound_ansatz = ansatz.bind_parameters(param_vals)
        expected_bound_ansatz = expected_ansatz.bind_parameters(param_vals)
        assert bound_ansatz == expected_bound_ansatz

        assert self.check_is_singlet(
            ansatz,
            param_vals,
            operator_mapping=bravyi_kitaev(n_spin_orbitals, n_electrons),
        )
