# Licensed under the MIT License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      https://mit-license.org/
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Iterator, Mapping, Sequence, TypeVar

from quri_parts.circuit import ImmutableQuantumCircuit

T = TypeVar("T")


class CircuitMapping(Mapping[ImmutableQuantumCircuit, T]):
    """Map an immutable quantum circuit to a value."""

    _vals_dictionary: dict[int, T]
    _keys_dictionary: dict[int, ImmutableQuantumCircuit]

    def _circuit_hash(self, circuit: ImmutableQuantumCircuit) -> int:
        """Hash the circuit gates and qubit count."""
        return hash((circuit.qubit_count, circuit.gates))

    def __init__(
        self, circuit_vals: Sequence[tuple[ImmutableQuantumCircuit, T]]
    ) -> None:
        self._vals_dictionary = {}
        self._keys_dictionary = {}
        for circuit, val in circuit_vals:
            self._vals_dictionary[self._circuit_hash(circuit)] = val
            self._keys_dictionary[self._circuit_hash(circuit)] = circuit

    def __getitem__(self, circuit: ImmutableQuantumCircuit) -> T:
        return self._vals_dictionary.__getitem__(self._circuit_hash(circuit))

    def __setitem__(self, circuit: ImmutableQuantumCircuit, value: T) -> None:
        self._vals_dictionary.__setitem__(self._circuit_hash(circuit), value)
        self._keys_dictionary.__setitem__(self._circuit_hash(circuit), circuit)

    def __delitem__(self, circuit: ImmutableQuantumCircuit) -> None:
        self._vals_dictionary.__delitem__(self._circuit_hash(circuit))
        self._keys_dictionary.__delitem__(self._circuit_hash(circuit))

    def __iter__(self) -> Iterator[ImmutableQuantumCircuit]:
        return self._keys_dictionary.values().__iter__()

    def __len__(self) -> int:
        return len(self._vals_dictionary)

    def __contains__(self, key: object) -> bool:
        return True if key in self._keys_dictionary.values() else False
