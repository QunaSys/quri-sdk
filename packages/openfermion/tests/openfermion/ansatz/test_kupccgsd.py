# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from quri_parts.circuit import LinearMappedUnboundParametricQuantumCircuit
from quri_parts.openfermion.ansatz.kupccgsd import (
    KUpCCGSD,
    _generalized_pair_double_excitations,
    _generalized_single_excitations,
)
from quri_parts.openfermion.transforms import symmetry_conserving_bravyi_kitaev


def test_generalized_single_excitations() -> None:
    n_spin_orbitals = 4
    s_excs = _generalized_single_excitations(n_spin_orbitals, 0)
    assert s_excs == [(0, 2), (1, 3), (2, 0), (3, 1)]
    s_excs = _generalized_single_excitations(n_spin_orbitals, 1)
    assert s_excs == [(1, 0), (1, 2), (3, 0), (3, 2)]

    n_spin_orbitals = 6
    s_excs = _generalized_single_excitations(n_spin_orbitals, 0)
    assert s_excs == [
        (0, 2),
        (0, 4),
        (1, 3),
        (1, 5),
        (2, 0),
        (2, 4),
        (3, 1),
        (3, 5),
        (4, 0),
        (4, 2),
        (5, 1),
        (5, 3),
    ]
    s_excs = _generalized_single_excitations(n_spin_orbitals, -1)
    assert s_excs == [
        (0, 1),
        (0, 3),
        (0, 5),
        (2, 1),
        (2, 3),
        (2, 5),
        (4, 1),
        (4, 3),
        (4, 5),
    ]


def test_generalized_pair_double_excitations() -> None:
    n_spin_orbitals = 4
    d_excs = _generalized_pair_double_excitations(n_spin_orbitals)
    assert d_excs == [(0, 1, 2, 3), (2, 3, 0, 1)]

    n_spin_orbitals = 6
    d_excs = _generalized_pair_double_excitations(n_spin_orbitals)
    assert d_excs == [
        (0, 1, 2, 3),
        (0, 1, 4, 5),
        (2, 3, 0, 1),
        (2, 3, 4, 5),
        (4, 5, 0, 1),
        (4, 5, 2, 3),
    ]


def test_kupccgsd() -> None:
    n_spin_orbitals = 4
    n_electrons = 2
    ansatz = KUpCCGSD(n_spin_orbitals, n_electrons)
    expected_ansatz = LinearMappedUnboundParametricQuantumCircuit(n_spin_orbitals)
    params = expected_ansatz.add_parameters(*[f"param{i}" for i in range(6)])
    expected_ansatz.add_ParametricPauliRotation_gate(
        (0, 1, 3, 2), (1, 1, 1, 2), {params[4]: -0.25}
    )
    expected_ansatz.add_ParametricPauliRotation_gate(
        (3, 0, 1, 2), (1, 2, 2, 2), {params[4]: 0.25}
    )
    expected_ansatz.add_ParametricPauliRotation_gate(
        (1, 3, 0, 2), (1, 1, 2, 1), {params[4]: 0.25}
    )
    expected_ansatz.add_ParametricPauliRotation_gate(
        (0, 3, 1, 2), (1, 1, 2, 1), {params[4]: 0.25}
    )
    expected_ansatz.add_ParametricPauliRotation_gate(
        (1, 0, 3, 2), (1, 2, 2, 2), {params[4]: -0.25}
    )
    expected_ansatz.add_ParametricPauliRotation_gate(
        (0, 3, 1, 2), (1, 2, 2, 2), {params[4]: -0.25}
    )
    expected_ansatz.add_ParametricPauliRotation_gate(
        (0, 1, 2, 3), (1, 1, 1, 2), {params[4]: -0.25}
    )
    expected_ansatz.add_ParametricPauliRotation_gate(
        (0, 1, 2, 3), (2, 2, 1, 2), {params[4]: 0.25}
    )
    expected_ansatz.add_ParametricPauliRotation_gate(
        (1, 3, 0, 2), (1, 1, 2, 1), {params[5]: -0.25}
    )
    expected_ansatz.add_ParametricPauliRotation_gate(
        (1, 0, 3, 2), (1, 2, 2, 2), {params[5]: 0.25}
    )
    expected_ansatz.add_ParametricPauliRotation_gate(
        (0, 1, 3, 2), (1, 1, 1, 2), {params[5]: 0.25}
    )
    expected_ansatz.add_ParametricPauliRotation_gate(
        (0, 1, 2, 3), (1, 1, 1, 2), {params[5]: 0.25}
    )
    expected_ansatz.add_ParametricPauliRotation_gate(
        (3, 0, 1, 2), (1, 2, 2, 2), {params[5]: -0.25}
    )
    expected_ansatz.add_ParametricPauliRotation_gate(
        (0, 1, 2, 3), (2, 2, 1, 2), {params[5]: -0.25}
    )
    expected_ansatz.add_ParametricPauliRotation_gate(
        (0, 3, 1, 2), (1, 1, 2, 1), {params[5]: -0.25}
    )
    expected_ansatz.add_ParametricPauliRotation_gate(
        (0, 3, 1, 2), (1, 2, 2, 2), {params[5]: 0.25}
    )
    expected_ansatz.add_ParametricPauliRotation_gate(
        (0, 1, 2), (2, 3, 1), {params[0]: -1.0}
    )
    expected_ansatz.add_ParametricPauliRotation_gate(
        (0, 1, 2), (1, 3, 2), {params[0]: 1.0}
    )
    expected_ansatz.add_ParametricPauliRotation_gate(
        (2, 1, 3), (3, 2, 1), {params[1]: -1.0}
    )
    expected_ansatz.add_ParametricPauliRotation_gate(
        (2, 1, 3), (3, 1, 2), {params[1]: 1.0}
    )
    expected_ansatz.add_ParametricPauliRotation_gate(
        (0, 1, 2), (2, 3, 1), {params[2]: 1.0}
    )
    expected_ansatz.add_ParametricPauliRotation_gate(
        (0, 1, 2), (1, 3, 2), {params[2]: -1.0}
    )
    expected_ansatz.add_ParametricPauliRotation_gate(
        (2, 1, 3), (3, 2, 1), {params[3]: 1.0}
    )
    expected_ansatz.add_ParametricPauliRotation_gate(
        (2, 1, 3), (3, 1, 2), {params[3]: -1.0}
    )
    assert ansatz.parameter_count == expected_ansatz.parameter_count
    assert ansatz._circuit.gates == expected_ansatz._circuit.gates
    param_vals = [0.1 * (i + 1) for i in range(ansatz.parameter_count)]
    bound_ansatz = ansatz.bind_parameters(param_vals)
    expected_bound_ansatz = expected_ansatz.bind_parameters(param_vals)
    assert bound_ansatz == expected_bound_ansatz

    n_spin_orbitals = 4
    n_electrons = 2
    k = 2
    ansatz = KUpCCGSD(
        n_spin_orbitals,
        n_electrons,
        k,
        fermion_qubit_mapping=symmetry_conserving_bravyi_kitaev,
        trotter_number=2,
    )
    expected_ansatz = LinearMappedUnboundParametricQuantumCircuit(n_spin_orbitals - 2)
    params = expected_ansatz.add_parameters(*[f"param{i}" for i in range(12)])
    for i in range(k):
        expected_ansatz.add_ParametricPauliRotation_gate(
            (0, 1), (1, 2), {params[4 + 6 * i]: 0.5}
        )
        expected_ansatz.add_ParametricPauliRotation_gate(
            (1, 0), (1, 2), {params[4 + 6 * i]: 0.5}
        )
        expected_ansatz.add_ParametricPauliRotation_gate(
            (0, 1), (1, 2), {params[5 + 6 * i]: -0.5}
        )
        expected_ansatz.add_ParametricPauliRotation_gate(
            (1, 0), (1, 2), {params[5 + 6 * i]: -0.5}
        )
        expected_ansatz.add_ParametricPauliRotation_gate(
            (0,), (2,), {params[0 + 6 * i]: -1.0}
        )
        expected_ansatz.add_ParametricPauliRotation_gate(
            (1,), (2,), {params[1 + 6 * i]: -1.0}
        )
        expected_ansatz.add_ParametricPauliRotation_gate(
            (0,), (2,), {params[2 + 6 * i]: 1.0}
        )
        expected_ansatz.add_ParametricPauliRotation_gate(
            (1,), (2,), {params[3 + 6 * i]: 1.0}
        )
        expected_ansatz.add_ParametricPauliRotation_gate(
            (0, 1), (1, 2), {params[4 + 6 * i]: 0.5}
        )
        expected_ansatz.add_ParametricPauliRotation_gate(
            (1, 0), (1, 2), {params[4 + 6 * i]: 0.5}
        )
        expected_ansatz.add_ParametricPauliRotation_gate(
            (0, 1), (1, 2), {params[5 + 6 * i]: -0.5}
        )
        expected_ansatz.add_ParametricPauliRotation_gate(
            (1, 0), (1, 2), {params[5 + 6 * i]: -0.5}
        )
        expected_ansatz.add_ParametricPauliRotation_gate(
            (0,), (2,), {params[0 + 6 * i]: -1.0}
        )
        expected_ansatz.add_ParametricPauliRotation_gate(
            (1,), (2,), {params[1 + 6 * i]: -1.0}
        )
        expected_ansatz.add_ParametricPauliRotation_gate(
            (0,), (2,), {params[2 + 6 * i]: 1.0}
        )
        expected_ansatz.add_ParametricPauliRotation_gate(
            (1,), (2,), {params[3 + 6 * i]: 1.0}
        )
    assert ansatz.parameter_count == expected_ansatz.parameter_count
    assert ansatz._circuit.gates == expected_ansatz._circuit.gates
    param_vals = [0.1 * (i + 1) for i in range(ansatz.parameter_count)]
    bound_ansatz = ansatz.bind_parameters(param_vals)
    expected_bound_ansatz = expected_ansatz.bind_parameters(param_vals)
    assert bound_ansatz == expected_bound_ansatz
