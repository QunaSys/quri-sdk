# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from quri_parts.qsub.op import Ident, Op
from quri_parts.qsub.opsub import ParamUnitarySubDef, param_opsub
from quri_parts.qsub.register import DEFAULT_QNAME, QRegSpec
from quri_parts.qsub.resolve import default_repository
from quri_parts.qsub.sub import Sub, SubBuilder

from . import NS
from .cnot import CNOT

SWAP = Op(Ident(NS, "SWAP"), 2, self_inverse=True, qregs=(QRegSpec(DEFAULT_QNAME, 2),))


def _swap_sub() -> Sub:
    b = SubBuilder(2)
    q0, q1 = b.qubits
    b.add_op(CNOT, (q0, q1))
    b.add_op(CNOT, (q1, q0))
    b.add_op(CNOT, (q0, q1))
    return b.build()


default_repository().register_sub(SWAP, _swap_sub())


class _Rot(ParamUnitarySubDef[int]):
    """Cyclically rotates the states of an ``n``-qubit register by one
    position.

    The rotation is implemented as a ladder of adjacent :data:`SWAP`
    operations, moving the state of the first qubit to the last qubit
    while shifting every other qubit one position toward the first.
    """

    name = "Rot"
    unitary = True

    def qubit_count_fn(self, n: int) -> int:
        return n

    def sub(self, builder: SubBuilder, n: int) -> None:
        qubits = builder.qubits
        for i in reversed(range(1, n)):
            builder.add_op(SWAP, (qubits[i - 1], qubits[i]))


Rot, RotSub = param_opsub(_Rot)
