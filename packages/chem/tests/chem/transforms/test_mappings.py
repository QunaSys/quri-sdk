# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from quri_parts.chem.transforms import (
    bravyi_kitaev_n_qubits_required,
    bravyi_kitaev_n_spin_orbitals,
    jordan_wigner_n_qubits_required,
    jordan_wigner_n_spin_orbitals,
    symmetry_conserving_bravyi_kitaev_n_qubits_required,
    symmetry_conserving_bravyi_kitaev_n_spin_orbitals,
)


def test_jordan_wigner_n_qubits_required() -> None:
    assert jordan_wigner_n_qubits_required(4) == 4
    assert jordan_wigner_n_qubits_required(6) == 6


def test_jordan_wigner_n_spin_orbitals() -> None:
    assert jordan_wigner_n_spin_orbitals(4) == 4
    assert jordan_wigner_n_spin_orbitals(6) == 6


def test_bravyi_kitaev_n_qubits_required() -> None:
    assert bravyi_kitaev_n_qubits_required(4) == 4
    assert bravyi_kitaev_n_qubits_required(6) == 6


def test_bravyi_kitaev_n_spin_orbitals() -> None:
    assert bravyi_kitaev_n_spin_orbitals(4) == 4
    assert bravyi_kitaev_n_spin_orbitals(6) == 6


def test_symmetry_conserving_bravyi_kitaev_n_qubits_required() -> None:
    assert symmetry_conserving_bravyi_kitaev_n_qubits_required(4) == 2
    assert symmetry_conserving_bravyi_kitaev_n_qubits_required(6) == 4


def test_symmetry_conserving_bravyi_kitaev_n_spin_orbitals() -> None:
    assert symmetry_conserving_bravyi_kitaev_n_spin_orbitals(4) == 6
    assert symmetry_conserving_bravyi_kitaev_n_spin_orbitals(6) == 8
