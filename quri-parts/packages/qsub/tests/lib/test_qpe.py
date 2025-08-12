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

from quri_parts.qsub.lib.qpe import QPE, LineH, QFTdag, QPEListUk
from quri_parts.qsub.lib.std import SWAP, Controlled, H, Phase
from quri_parts.qsub.resolve import default_repository

_repository = default_repository()


def test_qftdag() -> None:
    qftdag = QFTdag(4)
    resolver = _repository.find_resolver(qftdag)
    assert resolver is not None
    sub = resolver(qftdag, _repository)

    assert sub is not None
    assert len(sub.qubits) == 4
    assert len(sub.aux_qubits) == 0
    assert len(sub.registers) == 0
    assert len(sub.aux_registers) == 0
    q0, q1, q2, q3 = sub.qubits
    assert sub.operations == (
        (SWAP, (q0, q3), ()),
        (SWAP, (q1, q2), ()),
        (H, (q0,), ()),
        (Controlled(Phase(-pi / 2.0)), (q0, q1), ()),
        (H, (q1,), ()),
        (Controlled(Phase(-pi / 4.0)), (q0, q2), ()),
        (Controlled(Phase(-pi / 2.0)), (q1, q2), ()),
        (H, (q2,), ()),
        (Controlled(Phase(-pi / 8.0)), (q0, q3), ()),
        (Controlled(Phase(-pi / 4.0)), (q1, q3), ()),
        (Controlled(Phase(-pi / 2.0)), (q2, q3), ()),
        (H, (q3,), ()),
    )


def test_lineh() -> None:
    lineh = LineH(5)
    resolver = _repository.find_resolver(lineh)
    assert resolver is not None
    sub = resolver(lineh, _repository)

    assert sub is not None
    assert len(sub.qubits) == 5
    assert len(sub.aux_qubits) == 0
    assert len(sub.registers) == 0
    assert len(sub.aux_registers) == 0
    qs = sub.qubits
    assert sub.operations == (
        (H, (qs[0],), ()),
        (H, (qs[1],), ()),
        (H, (qs[2],), ()),
        (H, (qs[3],), ()),
        (H, (qs[4],), ()),
    )


def test_qpe() -> None:
    u = Phase(pi / 4.0)

    qpe = QPE(3, u)
    resolver = _repository.find_resolver(qpe)
    assert resolver is not None
    sub = resolver(qpe, _repository)

    assert sub is not None
    assert len(sub.qubits) == 4
    assert len(sub.aux_qubits) == 0
    assert len(sub.registers) == 0
    assert len(sub.aux_registers) == 0
    q0, q1, q2, q3 = sub.qubits
    assert sub.operations == (
        (LineH(3), (q0, q1, q2), ()),
        (Controlled(u), (q0, q3), ()),
        (Controlled(u), (q1, q3), ()),
        (Controlled(u), (q1, q3), ()),
        (Controlled(u), (q2, q3), ()),
        (Controlled(u), (q2, q3), ()),
        (Controlled(u), (q2, q3), ()),
        (Controlled(u), (q2, q3), ()),
        (QFTdag(3), (q0, q1, q2), ()),
    )


def test_qpe_list_uk() -> None:
    u1 = Phase(pi / 4.0)
    u2 = Phase(pi / 2.0)
    u4 = Phase(pi)
    u8 = Phase(2.0 * pi)

    qpe = QPEListUk(4, (u1, u2, u4, u8))
    resolver = _repository.find_resolver(qpe)
    assert resolver is not None
    sub = resolver(qpe, _repository)

    assert sub is not None
    assert len(sub.qubits) == 5
    assert len(sub.aux_qubits) == 0
    assert len(sub.registers) == 0
    assert len(sub.aux_registers) == 0
    pqs, sqs = sub.qubits[:4], sub.qubits[4:]
    assert sub.operations == (
        (LineH(4), pqs, ()),
        (Controlled(u1), (pqs[0], *sqs), ()),
        (Controlled(u2), (pqs[1], *sqs), ()),
        (Controlled(u4), (pqs[2], *sqs), ()),
        (Controlled(u8), (pqs[3], *sqs), ()),
        (QFTdag(4), pqs, ()),
    )
