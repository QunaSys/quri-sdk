# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from cmath import pi

from quri_parts.qsub.lib.std import SWAP, Controlled, H, Phase
from quri_parts.qsub.op import Op
from quri_parts.qsub.opsub import ParamUnitarySubDef, param_opsub
from quri_parts.qsub.sub import SubBuilder


class _QFTdag(ParamUnitarySubDef[int]):
    name = "QFTdag"

    def qubit_count_fn(self, bits: int) -> int:
        return bits

    def sub(self, builder: SubBuilder, bits: int) -> None:
        qubits = builder.qubits

        for k in range(bits // 2):
            builder.add_op(SWAP, (qubits[k], qubits[bits - k - 1]))

        for k in range(bits):
            for i in range(k):
                builder.add_op(
                    Controlled(Phase(-pi / (2 ** (k - i)))), (qubits[i], qubits[k])
                )
            builder.add_op(H, (qubits[k],))


QFTdag, QFTdagSub = param_opsub(_QFTdag)


class _LineH(ParamUnitarySubDef[int]):
    name = "LineH"

    def qubit_count_fn(self, bits: int) -> int:
        return bits

    def sub(self, builder: SubBuilder, bits: int) -> None:
        qubits = builder.qubits

        for k in range(bits):
            builder.add_op(H, (qubits[k],))


LineH, LineHSub = param_opsub(_LineH)


class _QPE(ParamUnitarySubDef[int, Op]):
    name = "QPE"

    def qubit_count_fn(self, bits: int, op: Op) -> int:
        return bits + op.qubit_count

    def sub(self, builder: SubBuilder, bits: int, op: Op) -> None:
        qubits = builder.qubits
        pqs, sqs = qubits[:bits], qubits[bits:]

        builder.add_op(LineH(bits), pqs)
        for k in range(bits):
            for _ in range(2**k):
                builder.add_op(Controlled(op), (pqs[k], *sqs))
        builder.add_op(QFTdag(bits), pqs)


#: Represents the Quantum Phase Estimation (QPE) circuit.
#:
#: It implements QPE by applying a line of H-gates, controlled unitaries and
#: the inverse quantum Fourier transform in the way described by e.g. Nielsen
#: and Chuang (2010) (https://doi.org/10.1017/CBO9780511976667).
#:
#: bits is the number of ancilla bits that are used to store the binary
#: expansion of the phase.
#:
#: op is the unitary operator that is applied, controlled sequentially by each
#: ancilla qubit in accordance with the standard formulation of QPE, where the
#: op is applied a number of times corresponding to the binary power
#: represented by its controlling ancilla bit.
QPE, QPESub = param_opsub(_QPE)


class _QPEListUk(ParamUnitarySubDef[int, tuple[Op]]):
    name = "QPEListUk"

    def qubit_count_fn(self, bits: int, ops: tuple[Op]) -> int:
        return bits + ops[0].qubit_count

    def sub(self, builder: SubBuilder, bits: int, ops: tuple[Op]) -> None:
        qubits = builder.qubits
        pqs, sqs = qubits[:bits], qubits[bits:]

        builder.add_op(LineH(bits), pqs)
        for k in range(bits):
            builder.add_op(Controlled(ops[k]), (pqs[k], *sqs))
        builder.add_op(QFTdag(bits), pqs)


#: Represents the Quantum Phase Estimation (QPE) circuit.
#:
#: It implements QPE by applying a line of H-gates, controlled unitaries and
#: the inverse quantum Fourier transform. The procedure provides a
#: generalization of the standard formulation described by e.g. Nielsen and
#: Chuang (2010) (https://doi.org/10.1017/CBO9780511976667).
#:
#: bits is the number of ancilla bits that are used to store the binary
#: expansion of the phase.
#:
#: ops is a sequence of unitary operators that are applied, controlled by each
#: ancilla bit in sequence. Each op in the sequence is applied only once.
QPEListUk, QPEListUkSub = param_opsub(_QPEListUk)
