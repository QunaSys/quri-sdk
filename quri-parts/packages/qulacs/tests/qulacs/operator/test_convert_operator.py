# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from qulacs import GeneralQuantumOperator

from quri_parts.core.operator import PAULI_IDENTITY, Operator, SinglePauli, pauli_label
from quri_parts.qulacs.operator import convert_operator


class TestConvertOperator:
    def test_convert_pauli_label(self) -> None:
        pauli = pauli_label("X5 Y9 Z17")
        qulacs_op = convert_operator(pauli, 18)

        assert isinstance(qulacs_op, GeneralQuantumOperator)
        assert qulacs_op.get_term_count() == 1
        qulacs_pauli = qulacs_op.get_term(0)
        converted_paulis = set(
            zip(qulacs_pauli.get_pauli_id_list(), qulacs_pauli.get_index_list())
        )
        assert converted_paulis == {
            (SinglePauli.X, 5),
            (SinglePauli.Y, 9),
            (SinglePauli.Z, 17),
        }
        assert qulacs_pauli.get_coef() == 1

    def test_convert_pauli_identity(self) -> None:
        qulacs_op = convert_operator(PAULI_IDENTITY, 3)

        assert isinstance(qulacs_op, GeneralQuantumOperator)
        assert qulacs_op.get_term_count() == 1
        qulacs_pauli = qulacs_op.get_term(0)
        assert qulacs_pauli.get_pauli_id_list() == []
        assert qulacs_pauli.get_coef() == 1

    def test_convert_operator(self) -> None:
        op = Operator(
            {
                pauli_label("X5 Y9 Z17"): 0.1,
                pauli_label("X3 Y8 Y12"): 0.2j,
                pauli_label("Z3 Y4 X10"): 0.3 + 0.4j,
                PAULI_IDENTITY: 0.5,
            }
        )
        qulacs_op = convert_operator(op, 18)

        assert isinstance(qulacs_op, GeneralQuantumOperator)
        assert qulacs_op.get_term_count() == 4
        terms = [qulacs_op.get_term(i) for i in range(4)]
        converted_paulis = [
            (set(zip(t.get_pauli_id_list(), t.get_index_list())), t.get_coef())
            for t in terms
        ]
        assert converted_paulis == [
            ({(SinglePauli.X, 5), (SinglePauli.Y, 9), (SinglePauli.Z, 17)}, 0.1),
            ({(SinglePauli.X, 3), (SinglePauli.Y, 8), (SinglePauli.Y, 12)}, 0.2j),
            ({(SinglePauli.Z, 3), (SinglePauli.Y, 4), (SinglePauli.X, 10)}, 0.3 + 0.4j),
            (set(), 0.5),
        ]
