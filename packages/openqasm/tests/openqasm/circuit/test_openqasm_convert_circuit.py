# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import io

from quri_parts.circuit import QuantumCircuit, gates
from quri_parts.openqasm.circuit import convert_gate_to_qasm_line, convert_to_qasm


class TestConvertGate:
    def test_x_gate(self) -> None:
        g = gates.X(123)
        qasm_expected = "x q[123];"
        assert convert_gate_to_qasm_line(g) == qasm_expected

    def test_y_gate(self) -> None:
        g = gates.Y(123)
        qasm_expected = "y q[123];"
        assert convert_gate_to_qasm_line(g) == qasm_expected

    def test_z_gate(self) -> None:
        g = gates.Z(123)
        qasm_expected = "z q[123];"
        assert convert_gate_to_qasm_line(g) == qasm_expected

    def test_h_gate(self) -> None:
        g = gates.H(123)
        qasm_expected = "h q[123];"
        assert convert_gate_to_qasm_line(g) == qasm_expected

    def test_s_gate(self) -> None:
        g = gates.S(123)
        qasm_expected = "s q[123];"
        assert convert_gate_to_qasm_line(g) == qasm_expected

    def test_sqrtx_gate(self) -> None:
        g = gates.SqrtX(123)
        qasm_expected = "sx q[123];"
        assert convert_gate_to_qasm_line(g) == qasm_expected

    def test_sdag_gate(self) -> None:
        g = gates.Sdag(123)
        qasm_expected = "sdg q[123];"
        assert convert_gate_to_qasm_line(g) == qasm_expected

    def test_t_gate(self) -> None:
        g = gates.T(123)
        qasm_expected = "t q[123];"
        assert convert_gate_to_qasm_line(g) == qasm_expected

    def test_tdag_gate(self) -> None:
        g = gates.Tdag(123)
        qasm_expected = "tdg q[123];"
        assert convert_gate_to_qasm_line(g) == qasm_expected

    def test_rx_gate(self) -> None:
        g = gates.RX(123, 0.5)
        qasm_expected = "rx(0.5) q[123];"
        assert convert_gate_to_qasm_line(g) == qasm_expected

    def test_ry_gate(self) -> None:
        g = gates.RY(123, -0.5)
        qasm_expected = "ry(-0.5) q[123];"
        assert convert_gate_to_qasm_line(g) == qasm_expected

    def test_rz_gate(self) -> None:
        g = gates.RZ(123, 0.5)
        qasm_expected = "rz(0.5) q[123];"
        assert convert_gate_to_qasm_line(g) == qasm_expected

    def test_cnot_gate(self) -> None:
        g = gates.CNOT(123, 456)
        qasm_expected = "cx q[123], q[456];"
        assert convert_gate_to_qasm_line(g) == qasm_expected

    def test_cz_gate(self) -> None:
        g = gates.CZ(123, 456)
        qasm_expected = "cz q[123], q[456];"
        assert convert_gate_to_qasm_line(g) == qasm_expected

    def test_swap_gate(self) -> None:
        g = gates.SWAP(123, 456)
        qasm_expected = "swap q[123], q[456];"
        assert convert_gate_to_qasm_line(g) == qasm_expected

    def test_toffoli_gate(self) -> None:
        g = gates.TOFFOLI(123, 456, 789)
        qasm_expected = "ccx q[123], q[456], q[789];"
        assert convert_gate_to_qasm_line(g) == qasm_expected

    def test_u1_gate(self) -> None:
        g = gates.U1(123, 1.5)
        qasm_expected = "u1(1.5) q[123];"
        assert convert_gate_to_qasm_line(g) == qasm_expected

    def test_u2_gate(self) -> None:
        g = gates.U2(123, 1.5, 2.5)
        qasm_expected = "u2(1.5, 2.5) q[123];"
        assert convert_gate_to_qasm_line(g) == qasm_expected

    def test_u3_gate(self) -> None:
        g = gates.U3(123, 1.5, 2.5, 3.5)
        qasm_expected = "u3(1.5, 2.5, 3.5) q[123];"
        assert convert_gate_to_qasm_line(g) == qasm_expected


class TestConvertToQasm:
    def test_convert_to_qasm(self) -> None:
        qubit_count = 7
        gate_list = [gates.X(0), gates.Z(6)]
        circuit = QuantumCircuit(qubit_count, gates=gate_list)
        str_io = io.StringIO()

        convert_to_qasm(circuit, str_io)
        actual = str_io.getvalue()

        expected = """OPENQASM 3;
include "stdgates.inc";
qubit[7] q;
bit[7] c;

x q[0];
z q[6];
c = measure q;"""

        assert actual == expected
