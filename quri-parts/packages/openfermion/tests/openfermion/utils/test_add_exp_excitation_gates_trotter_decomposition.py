# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from quri_parts.circuit import LinearMappedParametricQuantumCircuit
from quri_parts.core.circuit import add_parametric_commuting_paulis_exp_gate
from quri_parts.core.operator import Operator, pauli_label
from quri_parts.openfermion.transforms import (
    bravyi_kitaev,
    jordan_wigner,
    symmetry_conserving_bravyi_kitaev,
)
from quri_parts.openfermion.utils.add_exp_excitation_gates_trotter_decomposition import (  # noqa
    add_exp_excitation_gates_trotter_decomposition,
    create_anti_hermitian_sd_excitation_operator,
)


class TestCreateAntiHermitianSdExcitationOperator:
    def test_single_excitation_jw(self) -> None:
        jw_mapper = jordan_wigner.get_of_operator_mapper(n_spin_orbitals=4)

        s_exc = (0, 2)
        op = create_anti_hermitian_sd_excitation_operator(s_exc, jw_mapper)
        expected_op = Operator(
            {pauli_label("X0 Z1 Y2"): -0.5j, pauli_label("Y0 Z1 X2"): 0.5j}
        )
        assert op == expected_op

        s_exc = (1, 3)
        op = create_anti_hermitian_sd_excitation_operator(s_exc, jw_mapper)
        expected_op = Operator(
            {pauli_label("X1 Z2 Y3"): -0.5j, pauli_label("Y1 Z2 X3"): 0.5j}
        )
        assert op == expected_op

    def test_single_excitation_bk(self) -> None:
        bk_mapper = bravyi_kitaev.get_of_operator_mapper(n_spin_orbitals=4)

        s_exc = (0, 2)
        op = create_anti_hermitian_sd_excitation_operator(s_exc, bk_mapper)
        expected_op = Operator(
            {pauli_label("X0 Y1 X2"): 0.5j, pauli_label("Y0 Y1 Y2"): 0.5j}
        )
        assert op == expected_op

        s_exc = (1, 3)
        op = create_anti_hermitian_sd_excitation_operator(s_exc, bk_mapper)
        expected_op = Operator(
            {pauli_label("Z0 Y1 Z2"): 0.5j, pauli_label("Y1 Z3"): -0.5j}
        )
        assert op == expected_op

    def test_single_excitation_scbk(self) -> None:
        n_sorbs = 4
        n_elecs = 2
        scbk_mapper = symmetry_conserving_bravyi_kitaev.get_of_operator_mapper(
            n_sorbs, n_elecs, 0.0
        )

        s_exc = (0, 2)
        op = create_anti_hermitian_sd_excitation_operator(s_exc, scbk_mapper)
        expected_op = Operator({pauli_label("Y0"): 1.0j})
        assert op == expected_op

        s_exc = (1, 3)
        op = create_anti_hermitian_sd_excitation_operator(s_exc, scbk_mapper)
        expected_op = Operator({pauli_label("Y1"): 1.0j})
        assert op == expected_op

    def test_double_excitation_jw(self) -> None:
        jw_mapper = jordan_wigner.get_of_operator_mapper(n_spin_orbitals=4)

        d_exc = (0, 1, 2, 3)
        op = create_anti_hermitian_sd_excitation_operator(d_exc, jw_mapper)
        expected_op = Operator(
            {
                pauli_label("X0 X1 X2 Y3"): 0.125j,
                pauli_label("X0 X1 Y2 X3"): 0.125j,
                pauli_label("X0 Y1 X2 X3"): -0.125j,
                pauli_label("X0 Y1 Y2 Y3"): 0.125j,
                pauli_label("Y0 X1 X2 X3"): -0.125j,
                pauli_label("Y0 X1 Y2 Y3"): 0.125j,
                pauli_label("Y0 Y1 X2 Y3"): -0.125j,
                pauli_label("Y0 Y1 Y2 X3"): -0.125j,
            }
        )
        assert op == expected_op

        d_exc = (0, 3, 4, 7)
        op = create_anti_hermitian_sd_excitation_operator(d_exc, jw_mapper)
        expected_op = Operator(
            {
                pauli_label("X0 Z1 Z2 X3 X4 Z5 Z6 Y7"): 0.125j,
                pauli_label("X0 Z1 Z2 X3 Y4 Z5 Z6 X7"): 0.125j,
                pauli_label("X0 Z1 Z2 Y3 X4 Z5 Z6 X7"): -0.125j,
                pauli_label("X0 Z1 Z2 Y3 Y4 Z5 Z6 Y7"): 0.125j,
                pauli_label("Y0 Z1 Z2 X3 X4 Z5 Z6 X7"): -0.125j,
                pauli_label("Y0 Z1 Z2 X3 Y4 Z5 Z6 Y7"): 0.125j,
                pauli_label("Y0 Z1 Z2 Y3 X4 Z5 Z6 Y7"): -0.125j,
                pauli_label("Y0 Z1 Z2 Y3 Y4 Z5 Z6 X7"): -0.125j,
            }
        )
        assert op == expected_op

    def test_double_excitation_bk(self) -> None:
        bk_mapper = bravyi_kitaev.get_of_operator_mapper(n_spin_orbitals=8)

        d_exc = (0, 1, 2, 3)
        op = create_anti_hermitian_sd_excitation_operator(d_exc, bk_mapper)
        expected_op = Operator(
            {
                pauli_label("X0 Y2"): 0.125j,
                pauli_label("X0 Z1 Y2"): 0.125j,
                pauli_label("Y0 X2"): -0.125j,
                pauli_label("Y0 Z1 X2"): -0.125j,
                pauli_label("Y0 Z1 X2 Z3"): -0.125j,
                pauli_label("Y0 X2 Z3"): -0.125j,
                pauli_label("X0 Z1 Y2 Z3"): 0.125j,
                pauli_label("X0 Y2 Z3"): 0.125j,
            }
        )
        assert op == expected_op

        d_exc = (0, 3, 4, 7)
        op = create_anti_hermitian_sd_excitation_operator(d_exc, bk_mapper)
        expected_op = Operator(
            {
                pauli_label("Y0 Y1 Z2 X4 Y5 Z6"): -0.125j,
                pauli_label("X0 X1 Z3 X4 Y5 Z6"): 0.125j,
                pauli_label("X0 Y1 Z2 Y4 Y5 Z6"): 0.125j,
                pauli_label("Y0 X1 Z3 Y4 Y5 Z6"): 0.125j,
                pauli_label("X0 Y1 Z2 Z3 X4 X5 Z7"): -0.125j,
                pauli_label("Y0 X1 X4 X5 Z7"): -0.125j,
                pauli_label("Y0 Y1 Z2 Z3 Y4 X5 Z7"): -0.125j,
                pauli_label("X0 X1 Y4 X5 Z7"): 0.125j,
            }
        )
        assert op == expected_op

    def test_double_excitation_scbk(self) -> None:
        scbk_mapper = symmetry_conserving_bravyi_kitaev.get_of_operator_mapper(
            4, 2, 0.0
        )

        d_exc = (0, 1, 2, 3)
        op = create_anti_hermitian_sd_excitation_operator(d_exc, scbk_mapper)
        expected_op = Operator(
            {
                pauli_label("X0 Y1"): -0.5j,
                pauli_label("Y0 X1"): -0.5j,
            }
        )
        assert op == expected_op

        scbk_mapper = symmetry_conserving_bravyi_kitaev.get_of_operator_mapper(
            8, 2, 0.0
        )
        d_exc = (0, 3, 4, 7)
        op = create_anti_hermitian_sd_excitation_operator(d_exc, scbk_mapper)
        expected_op = Operator(
            {
                pauli_label("Y0 Y1 X2 Z3 Y4 Z5"): 0.125j,
                pauli_label("X0 Y1 X2 X4 Z5"): -0.125j,
                pauli_label("X0 Y1 Y2 Z3 Y4 Z5"): -0.125j,
                pauli_label("Y0 Y1 Y2 X4 Z5"): -0.125j,
                pauli_label("X0 Y1 X2 Z3 X4"): -0.125j,
                pauli_label("Y0 Y1 X2 Y4"): 0.125j,
                pauli_label("Y0 Y1 Y2 Z3 X4"): -0.125j,
                pauli_label("X0 Y1 Y2 Y4"): -0.125j,
            }
        )
        assert op == expected_op


class TestAddExpExcitationGatesTrotterDecomposition:
    def test_trotter1_jw(self) -> None:
        n_spin_orbitals = 4
        excitation_indices = [(0, 2)]
        jw_mapper = jordan_wigner.get_of_operator_mapper(n_spin_orbitals)
        trotter_number = 1

        circuit = LinearMappedParametricQuantumCircuit(n_spin_orbitals)
        param = circuit.add_parameter("param")
        add_exp_excitation_gates_trotter_decomposition(
            circuit, excitation_indices, [param], jw_mapper, coef=1 / trotter_number
        )
        expected_circuit = LinearMappedParametricQuantumCircuit(n_spin_orbitals)
        param = expected_circuit.add_parameter("param")
        expected_circuit.add_ParametricPauliRotation_gate(
            (0, 1, 2), (2, 3, 1), {param: -1.0}
        )
        expected_circuit.add_ParametricPauliRotation_gate(
            (0, 1, 2), (1, 3, 2), {param: 1.0}
        )
        assert circuit.parameter_count == expected_circuit.parameter_count
        assert circuit._circuit.gates == expected_circuit._circuit.gates
        param_vals = [0.1 * (i + 1) for i in range(circuit.parameter_count)]
        bound_circuit = circuit.bind_parameters(param_vals)
        expected_bound_circuit = expected_circuit.bind_parameters(param_vals)
        assert bound_circuit == expected_bound_circuit

    def test_trotter2_jw(self) -> None:
        n_spin_orbitals = 4
        excitation_indices = [(0, 2)]
        jw_mapper = jordan_wigner.get_of_operator_mapper(n_spin_orbitals)
        trotter_number = 2

        circuit = LinearMappedParametricQuantumCircuit(n_spin_orbitals)
        param = circuit.add_parameter("param")
        add_exp_excitation_gates_trotter_decomposition(
            circuit, excitation_indices, [param], jw_mapper, coef=1 / trotter_number
        )
        expected_circuit = LinearMappedParametricQuantumCircuit(n_spin_orbitals)
        param = expected_circuit.add_parameter("param")
        expected_circuit.add_ParametricPauliRotation_gate(
            (0, 1, 2), (2, 3, 1), {param: -0.5}
        )
        expected_circuit.add_ParametricPauliRotation_gate(
            (0, 1, 2), (1, 3, 2), {param: 0.5}
        )
        assert circuit.parameter_count == expected_circuit.parameter_count
        assert circuit._circuit.gates == expected_circuit._circuit.gates
        param_vals = [0.1 * (i + 1) for i in range(circuit.parameter_count)]
        bound_circuit = circuit.bind_parameters(param_vals)
        expected_bound_circuit = expected_circuit.bind_parameters(param_vals)
        assert bound_circuit == expected_bound_circuit

    def test_trotter1_bk(self) -> None:
        n_spin_orbitals = 4
        excitation_indices = [(0, 2)]
        bk_mapper = bravyi_kitaev.get_of_operator_mapper(n_spin_orbitals)
        trotter_number = 1

        circuit = LinearMappedParametricQuantumCircuit(n_spin_orbitals)
        param = circuit.add_parameter("param")
        add_exp_excitation_gates_trotter_decomposition(
            circuit, excitation_indices, [param], bk_mapper, coef=1 / trotter_number
        )
        expected_circuit = LinearMappedParametricQuantumCircuit(n_spin_orbitals)
        param = expected_circuit.add_parameter("param")
        expected_circuit.add_ParametricPauliRotation_gate(
            (0, 1, 2), (1, 2, 1), {param: -1.0}
        )
        expected_circuit.add_ParametricPauliRotation_gate(
            (0, 1, 2), (2, 2, 2), {param: -1.0}
        )
        assert circuit.parameter_count == expected_circuit.parameter_count
        assert circuit._circuit.gates == expected_circuit._circuit.gates
        param_vals = [0.1 * (i + 1) for i in range(circuit.parameter_count)]
        bound_circuit = circuit.bind_parameters(param_vals)
        expected_bound_circuit = expected_circuit.bind_parameters(param_vals)
        assert bound_circuit == expected_bound_circuit

    def test_trotter1_scbk(self) -> None:
        n_spin_orbitals = 4
        excitation_indices = [(0, 2)]
        scbk_mapper = symmetry_conserving_bravyi_kitaev.get_of_operator_mapper(
            n_spin_orbitals, 2, 0.0
        )
        trotter_number = 1

        n_qubits = symmetry_conserving_bravyi_kitaev.n_qubits_required(n_spin_orbitals)
        circuit = LinearMappedParametricQuantumCircuit(n_qubits)
        param = circuit.add_parameter("param")
        add_exp_excitation_gates_trotter_decomposition(
            circuit, excitation_indices, [param], scbk_mapper, coef=1 / trotter_number
        )
        expected_circuit = LinearMappedParametricQuantumCircuit(n_qubits)
        param = expected_circuit.add_parameter("param")
        expected_circuit.add_ParametricPauliRotation_gate((0,), (2,), {param: -2.0})
        assert circuit.parameter_count == expected_circuit.parameter_count
        assert circuit._circuit.gates == expected_circuit._circuit.gates
        param_vals = [0.1 * (i + 1) for i in range(circuit.parameter_count)]
        bound_circuit = circuit.bind_parameters(param_vals)
        expected_bound_circuit = expected_circuit.bind_parameters(param_vals)
        assert bound_circuit == expected_bound_circuit

    def test_doubles(self) -> None:
        n_spin_orbitals = 6
        excitation_indices = [(0, 1, 4, 5)]
        jw_mapper = jordan_wigner.get_of_operator_mapper(n_spin_orbitals)
        trotter_number = 1

        circuit = LinearMappedParametricQuantumCircuit(n_spin_orbitals)
        param = circuit.add_parameter("param")
        add_exp_excitation_gates_trotter_decomposition(
            circuit, excitation_indices, [param], jw_mapper, coef=1 / trotter_number
        )
        expected_circuit = LinearMappedParametricQuantumCircuit(n_spin_orbitals)
        param = expected_circuit.add_parameter("param")
        expected_circuit.add_ParametricPauliRotation_gate(
            (0, 1, 5, 4), (1, 1, 1, 2), {param: -0.25}
        )
        expected_circuit.add_ParametricPauliRotation_gate(
            (0, 1, 5, 4), (2, 2, 1, 2), {param: 0.25}
        )
        expected_circuit.add_ParametricPauliRotation_gate(
            (1, 0, 4, 5), (1, 2, 1, 1), {param: 0.25}
        )
        expected_circuit.add_ParametricPauliRotation_gate(
            (0, 1, 4, 5), (1, 2, 1, 1), {param: 0.25}
        )
        expected_circuit.add_ParametricPauliRotation_gate(
            (1, 0, 4, 5), (1, 2, 2, 2), {param: -0.25}
        )
        expected_circuit.add_ParametricPauliRotation_gate(
            (0, 1, 4, 5), (1, 2, 2, 2), {param: -0.25}
        )
        expected_circuit.add_ParametricPauliRotation_gate(
            (0, 1, 4, 5), (1, 1, 1, 2), {param: -0.25}
        )
        expected_circuit.add_ParametricPauliRotation_gate(
            (0, 1, 4, 5), (2, 2, 1, 2), {param: 0.25}
        )
        assert circuit.parameter_count == expected_circuit.parameter_count
        assert circuit._circuit.gates == expected_circuit._circuit.gates
        param_vals = [0.1 * (i + 1) for i in range(circuit.parameter_count)]
        bound_circuit = circuit.bind_parameters(param_vals)
        expected_bound_circuit = expected_circuit.bind_parameters(param_vals)
        assert bound_circuit == expected_bound_circuit


class TestAddExpPauliGatesFromLinearMappedFunctions:
    """Tests if the `add_parametric_commuting_paulis_exp_gate` function adds
    the operator created by `create_anti_hermitian_sd_excitation_operator` to
    the circuit properly."""

    def test_jw(self) -> None:
        n_spin_orbitals = 6
        jw_mapper = jordan_wigner.get_of_operator_mapper(n_spin_orbitals)

        circuit = LinearMappedParametricQuantumCircuit(n_spin_orbitals)
        param1 = circuit.add_parameter("param1")
        param2 = circuit.add_parameter("param2")

        operator_1 = (
            create_anti_hermitian_sd_excitation_operator((0, 1, 2, 3), jw_mapper) * -1j
        )
        add_parametric_commuting_paulis_exp_gate(
            circuit, {param1: 4}, operator_1, coeff=0.5
        )
        operator_2 = (
            create_anti_hermitian_sd_excitation_operator((2, 3, 4, 5), jw_mapper) * -1j
        )
        add_parametric_commuting_paulis_exp_gate(
            circuit, {param2: -2}, operator_2, coeff=0.4
        )
        operator_3 = (
            create_anti_hermitian_sd_excitation_operator((0, 1, 3, 4), jw_mapper) * -1j
        )
        add_parametric_commuting_paulis_exp_gate(
            circuit, {param1: 1, param2: -3}, operator_3, coeff=-2.0
        )

        expected_circuit = LinearMappedParametricQuantumCircuit(n_spin_orbitals)
        param1 = expected_circuit.add_parameter("param1")
        param2 = expected_circuit.add_parameter("param2")

        # param 1
        expected_circuit.add_ParametricPauliRotation_gate(
            (0, 1, 3, 2), (1, 1, 1, 2), {param1: -0.5}
        )
        expected_circuit.add_ParametricPauliRotation_gate(
            (3, 0, 1, 2), (1, 2, 2, 2), {param1: 0.5}
        )
        expected_circuit.add_ParametricPauliRotation_gate(
            (1, 3, 0, 2), (1, 1, 2, 1), {param1: 0.5}
        )
        expected_circuit.add_ParametricPauliRotation_gate(
            (0, 3, 1, 2), (1, 1, 2, 1), {param1: 0.5}
        )
        expected_circuit.add_ParametricPauliRotation_gate(
            (1, 0, 3, 2), (1, 2, 2, 2), {param1: -0.5}
        )
        expected_circuit.add_ParametricPauliRotation_gate(
            (0, 3, 1, 2), (1, 2, 2, 2), {param1: -0.5}
        )
        expected_circuit.add_ParametricPauliRotation_gate(
            (0, 1, 2, 3), (1, 1, 1, 2), {param1: -0.5}
        )
        expected_circuit.add_ParametricPauliRotation_gate(
            (0, 1, 2, 3), (2, 2, 1, 2), {param1: 0.5}
        )

        # param 2
        expected_circuit.add_ParametricPauliRotation_gate(
            (3, 5, 2, 4), (1, 1, 1, 2), {param2: 0.2}
        )
        expected_circuit.add_ParametricPauliRotation_gate(
            (3, 5, 4, 2), (2, 1, 2, 2), {param2: -0.2}
        )
        expected_circuit.add_ParametricPauliRotation_gate(
            (3, 4, 5, 2), (1, 1, 1, 2), {param2: -0.2}
        )
        expected_circuit.add_ParametricPauliRotation_gate(
            (5, 3, 4, 2), (1, 2, 1, 1), {param2: -0.2}
        )
        expected_circuit.add_ParametricPauliRotation_gate(
            (3, 5, 4, 2), (1, 2, 2, 2), {param2: 0.2}
        )
        expected_circuit.add_ParametricPauliRotation_gate(
            (3, 2, 4, 5), (2, 1, 2, 2), {param2: 0.2}
        )
        expected_circuit.add_ParametricPauliRotation_gate(
            (3, 4, 2, 5), (1, 1, 1, 2), {param2: 0.2}
        )
        expected_circuit.add_ParametricPauliRotation_gate(
            (5, 3, 4, 2), (2, 2, 1, 2), {param2: -0.2}
        )

        # param1, param 2
        expected_circuit.add_ParametricPauliRotation_gate(
            (0, 1, 4, 3), (1, 1, 1, 2), {param1: 1 * 0.5, param2: -3 * 0.5}
        )
        expected_circuit.add_ParametricPauliRotation_gate(
            (0, 1, 4, 3), (2, 2, 1, 2), {param1: -1 * 0.5, param2: 3 * 0.5}
        )
        expected_circuit.add_ParametricPauliRotation_gate(
            (1, 0, 4, 3), (1, 2, 1, 1), {param1: -1 * 0.5, param2: 3 * 0.5}
        )
        expected_circuit.add_ParametricPauliRotation_gate(
            (0, 1, 4, 3), (1, 2, 1, 1), {param1: -1 * 0.5, param2: 3 * 0.5}
        )
        expected_circuit.add_ParametricPauliRotation_gate(
            (1, 0, 3, 4), (1, 2, 2, 2), {param1: 1 * 0.5, param2: -3 * 0.5}
        )
        expected_circuit.add_ParametricPauliRotation_gate(
            (0, 3, 1, 4), (1, 2, 2, 2), {param1: 1 * 0.5, param2: -3 * 0.5}
        )
        expected_circuit.add_ParametricPauliRotation_gate(
            (0, 1, 3, 4), (1, 1, 1, 2), {param1: 1 * 0.5, param2: -3 * 0.5}
        )
        expected_circuit.add_ParametricPauliRotation_gate(
            (3, 0, 1, 4), (1, 2, 2, 2), {param1: -1 * 0.5, param2: 3 * 0.5}
        )

        assert circuit.parameter_count == expected_circuit.parameter_count
        assert circuit._circuit.gates == expected_circuit._circuit.gates
        param_vals = [0.1 * (i + 1) for i in range(circuit.parameter_count)]
        bound_circuit = circuit.bind_parameters(param_vals)
        expected_bound_circuit = expected_circuit.bind_parameters(param_vals)
        assert set(bound_circuit.gates) == set(expected_bound_circuit.gates)
