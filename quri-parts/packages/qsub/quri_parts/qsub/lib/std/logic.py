# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from contextlib import contextmanager
from typing import Generator

from quri_parts.qsub.qubit import Qubit
from quri_parts.qsub.sub import SubBuilder

from .cnot import CNOT
from .conditional import conditional
from .cz import CZ
from .measure import M
from .single_clifford import H, Sdag
from .t import T, Tdag
from .toffoli import Toffoli


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
