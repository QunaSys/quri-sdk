# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from contextlib import AbstractContextManager, contextmanager
from typing import Callable, Generator, Optional

from typing_extensions import TypeAlias

from quri_parts.qsub.op import Op
from quri_parts.qsub.qubit import Qubit
from quri_parts.qsub.resolve import default_repository
from quri_parts.qsub.sub import Sub, SubBuilder

from .cnot import CNOT
from .conditional import conditional
from .control import Controlled, MultiControlled
from .cz import CZ
from .measure import M
from .single_clifford import H, Sdag, X
from .t import T, Tdag
from .toffoli import Toffoli

_ScopedAnd: TypeAlias = Callable[
    [SubBuilder, Qubit, Qubit], AbstractContextManager[Qubit]
]


@contextmanager
def scoped_and(
    builder: SubBuilder, i0: Qubit, i1: Qubit
) -> Generator[Qubit, None, None]:
    a = builder.add_aux_qubit()
    builder.add_op(Toffoli, (i0, i1, a))
    yield a
    builder.add_op(Toffoli, (i0, i1, a))


# https://arxiv.org/abs/1709.06648
# https://arxiv.org/abs/1805.03662
@contextmanager
def scoped_and_clifford_t(
    builder: SubBuilder, i0: Qubit, i1: Qubit
) -> Generator[Qubit, None, None]:
    a = builder.add_aux_qubit()  # a is a qubit
    builder.add_op(H, (a,))
    builder.add_op(T, (a,))
    builder.add_op(CNOT, (i1, a))
    builder.add_op(Tdag, (a,))
    builder.add_op(CNOT, (i0, a))
    builder.add_op(T, (a,))
    builder.add_op(CNOT, (i1, a))
    builder.add_op(Tdag, (a,))
    builder.add_op(H, (a,))
    builder.add_op(Sdag, (a,))
    yield a
    builder.add_op(H, (a,))
    r = builder.add_aux_register()
    builder.add_op(M, (a,), (r,))
    with conditional(builder, r):
        builder.add_op(CZ, (i0, i1))


@contextmanager
def scoped_and_single_toffoli(
    builder: SubBuilder, i0: Qubit, i1: Qubit
) -> Generator[Qubit, None, None]:
    a = builder.add_aux_qubit()
    builder.add_op(Toffoli, (i0, i1, a))
    yield a
    builder.add_op(H, (a,))
    r = builder.add_aux_register()
    builder.add_op(M, (a,), (r,))
    with conditional(builder, r):
        builder.add_op(CZ, (i0, i1))


def _multi_controlled_sub(
    op: Op, control_bits: int, control_value: int, s_and: _ScopedAnd
) -> Sub:
    builder = SubBuilder(op.qubit_count + control_bits)
    qubits = builder.qubits

    if not control_bits >= 1:
        raise ValueError(
            f"control_bits should be a positive integer but specified: {control_bits}"
        )

    neg_qubits = [i for i in range(control_bits) if ((control_value >> i) & 1) == 0]

    def add_neg() -> None:
        for i in neg_qubits:
            builder.add_op(X, (qubits[i],))

    if control_bits == 1:
        add_neg()
        builder.add_op(Controlled(op), (qubits[0], qubits[1]))
        add_neg()
        return builder.build()

    def recursive_and(i: int, a: Optional[Qubit] = None) -> None:
        if i == control_bits - 1:
            assert a
            builder.add_op(Controlled(op), (a, *qubits[control_bits:]))
        else:
            if i == 0:
                i0 = qubits[0]
            else:
                assert a
                i0 = a
            i1 = qubits[i + 1]
            with s_and(builder, i0, i1) as a:
                recursive_and(i + 1, a)

    add_neg()
    recursive_and(0)
    add_neg()

    return builder.build()


def MultiControlledSub(op: Op, control_bits: int, control_value: int) -> Sub:
    return _multi_controlled_sub(op, control_bits, control_value, scoped_and)


def MultiControlledCliffordTSub(op: Op, control_bits: int, control_value: int) -> Sub:
    return _multi_controlled_sub(op, control_bits, control_value, scoped_and_clifford_t)


def MultiControlledSingleToffoliSub(
    op: Op, control_bits: int, control_value: int
) -> Sub:
    return _multi_controlled_sub(
        op, control_bits, control_value, scoped_and_single_toffoli
    )


_repo = default_repository()
_repo.register_sub(MultiControlled, MultiControlledSub)
