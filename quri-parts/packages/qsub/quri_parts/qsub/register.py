# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass
from typing import NamedTuple, Sequence, overload

from .qubit import Qubit

CTRL_QNAME = "ctrl"
DEFAULT_QNAME = "qs"
TARGET_QNAME = "target"


class Register(NamedTuple):
    uid: int

    def __str__(self) -> str:
        return f"r{self.uid}"


class QRegSpec(NamedTuple):
    name: str
    qubit_count: int


@dataclass(frozen=True)
class QuantumRegister:
    name: str
    qubits: Sequence[Qubit]

    @property
    def size(self) -> int:
        return len(self.qubits)

    @overload
    def __getitem__(self, idx: int) -> Qubit:
        ...

    @overload
    def __getitem__(self, idx: slice) -> Sequence[Qubit]:
        ...

    def __getitem__(self, idx: int | slice) -> Qubit | Sequence[Qubit]:
        return self.qubits[idx]

    def get_qubit_idx(self, qubit: Qubit) -> int:
        return self.qubits.index(qubit)

    def __str__(self) -> str:
        qstr = ", ".join([str(q) for q in self.qubits])
        return f"{self.name}<{qstr}>"


def check_register_appear_once(qregs: Sequence[QRegSpec]) -> None:
    name_set: set[str] = set()
    for qr in qregs:
        if qr.name in name_set:
            raise ValueError(f"Duplicated register of name {qr.name}")
        name_set.add(qr.name)


def get_default_qreg_sequence(qubit_count: int) -> Sequence[QRegSpec]:
    return tuple(QRegSpec(f"{DEFAULT_QNAME}_{i}", 1) for i in range(qubit_count))


def get_qregs_from_quantum_register_map(
    qr_map: dict[str, QuantumRegister]
) -> Sequence[QRegSpec]:
    return tuple(QRegSpec(name, len(qr.qubits)) for name, qr in qr_map.items())
