# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from quri_parts.qsub.lib import std
from quri_parts.qsub.opsub import ParamUnitarySubDef, param_opsub
from quri_parts.qsub.sub import SubBuilder

from . import NS


class _Pauli(ParamUnitarySubDef[tuple[int, ...]]):
    ns = NS
    name = "Pauli"

    def qubit_count_fn(self, pauli_ids: tuple[int, ...]) -> int:
        return len(pauli_ids)

    def sub(self, builder: SubBuilder, pauli_ids: tuple[int, ...]) -> None:
        for qubit, pauli in zip(builder.qubits, pauli_ids):
            match pauli:
                case 1:
                    builder.add_op(std.X, (qubit,))
                case 2:
                    builder.add_op(std.Y, (qubit,))
                case 3:
                    builder.add_op(std.Z, (qubit,))
                case _:
                    raise ValueError(f"Unknown pauli id: {pauli}")


Pauli, PauliSub = param_opsub(_Pauli)


class _PauliRotation(ParamUnitarySubDef[tuple[int, ...], float]):
    ns = NS
    name = "PauliRotation"

    def qubit_count_fn(self, pauli_ids: tuple[int, ...], angle: float) -> int:
        return len(pauli_ids)

    def sub(
        self, builder: SubBuilder, pauli_ids: tuple[int, ...], angle: float
    ) -> None:
        qubits = builder.qubits

        def add_rot_gates(rot_sign: int) -> None:
            for qubit, pauli in zip(qubits, pauli_ids):
                match pauli:
                    case 1:
                        builder.add_op(std.H, (qubit,))
                    case 2:
                        op = std.SqrtX if rot_sign == 1 else std.SqrtXdag
                        builder.add_op(op, (qubit,))
                    case 3:
                        pass
                    case _:
                        raise ValueError(f"Unknown pauli id: {pauli}")

        add_rot_gates(1)
        for qubit in reversed(qubits[1:]):
            builder.add_op(std.CNOT, (qubit, qubits[0]))
        builder.add_op(std.RZ(angle), (qubits[0],))
        for qubit in qubits[1:]:
            builder.add_op(std.CNOT, (qubit, qubits[0]))
        add_rot_gates(-1)


PauliRotation, PauliRotationSub = param_opsub(_PauliRotation)
