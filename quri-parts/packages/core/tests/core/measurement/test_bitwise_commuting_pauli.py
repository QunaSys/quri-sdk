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

from quri_parts.circuit import H, Sdag
from quri_parts.core.measurement import (
    bitwise_commuting_pauli_measurement_circuit,
    bitwise_pauli_reconstructor_factory,
)
from quri_parts.core.operator import PAULI_IDENTITY, pauli_label


class TestBitwiseCommutingPauliMeasurementCircuit:
    def test_raise_when_pauli_set_empty(self) -> None:
        with pytest.raises(ValueError):
            bitwise_commuting_pauli_measurement_circuit(set())

    def test_raise_when_non_bitwise_commuting_paulis_exist(self) -> None:
        with pytest.raises(ValueError):
            bitwise_commuting_pauli_measurement_circuit(
                set(
                    {
                        pauli_label("Z0"),
                        pauli_label("Z0 X1 Y2"),
                        pauli_label("Z0 Y1 X2"),
                    }
                ),
            )

    def test_generate_correct_circuit(self) -> None:
        pauli_set = set(
            {
                pauli_label("X1"),
                pauli_label("X1 Y2"),
                pauli_label("Z3"),
            }
        )
        circuit = bitwise_commuting_pauli_measurement_circuit(pauli_set)

        assert len(circuit) == 3
        assert circuit[0] == H(1)
        assert circuit[1] == Sdag(2)
        assert circuit[2] == H(2)


class TestBitwisePauliReconstructor:
    def test_return_one_for_identity(self) -> None:
        reconstructor = bitwise_pauli_reconstructor_factory(PAULI_IDENTITY)
        assert reconstructor(0) == 1

    def test_reconstruct_correctly(self) -> None:
        reconstructor = bitwise_pauli_reconstructor_factory(pauli_label("X2 Y4 Z6"))

        values = {
            0: 1,  # no bit set
            0b100: -1,  # bit set on X2
            0b11100: 1,  # bit set on X2, Y4
            0b1001100: 1,  # bit set on X2, Z6
            0b1001010101: -1,  # bit set on X2, Y4, Z6
        }

        for bits, expected in values.items():
            assert reconstructor(bits) == expected
