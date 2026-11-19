# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest

from quri_parts.chem.ansatz.all_singles_doubles import AllSinglesDoubles
from quri_parts.chem.utils.excitations import (
    add_double_excitation_circuit,
    add_single_excitation_circuit,
    excitations,
)
from quri_parts.circuit import LinearMappedParametricQuantumCircuit


def test_all_singles_doubles() -> None:
    qubit_count = 4
    n_electrons = 2
    circuit = AllSinglesDoubles(qubit_count, n_electrons, sz=0)
    expected_circuit = LinearMappedParametricQuantumCircuit(qubit_count)
    s_exc_indices, d_exc_indices = excitations(qubit_count, n_electrons)
    s_exc_params = expected_circuit.add_parameters(
        *[f"theta_s_{i}" for i in range(len(s_exc_indices))]
    )
    d_exc_params = expected_circuit.add_parameters(
        *[f"phi_d_{i}" for i in range(len(d_exc_indices))]
    )
    for d_exc, d_exc_param in zip(d_exc_indices, d_exc_params):
        add_double_excitation_circuit(expected_circuit, d_exc, d_exc_param)
    for s_exc, s_exc_param in zip(s_exc_indices, s_exc_params):
        add_single_excitation_circuit(expected_circuit, s_exc, s_exc_param)
    assert circuit.parameter_count == expected_circuit.parameter_count
    assert circuit._circuit.gates == expected_circuit._circuit.gates
    params = [0.1 * (i + 1) for i in range(circuit.parameter_count)]
    bound_circuit = circuit.bind_parameters(params)
    expected_bound_circuit = expected_circuit.bind_parameters(params)
    assert bound_circuit == expected_bound_circuit

    qubit_count = 8
    n_electrons = 4
    circuit = AllSinglesDoubles(qubit_count, n_electrons, sz=0)
    expected_circuit = LinearMappedParametricQuantumCircuit(qubit_count)
    s_exc_indices, d_exc_indices = excitations(qubit_count, n_electrons)
    s_exc_params = expected_circuit.add_parameters(
        *[f"theta_s_{i}" for i in range(len(s_exc_indices))]
    )
    d_exc_params = expected_circuit.add_parameters(
        *[f"phi_d_{i}" for i in range(len(d_exc_indices))]
    )
    for d_exc, d_exc_param in zip(d_exc_indices, d_exc_params):
        add_double_excitation_circuit(expected_circuit, d_exc, d_exc_param)
    for s_exc, s_exc_param in zip(s_exc_indices, s_exc_params):
        add_single_excitation_circuit(expected_circuit, s_exc, s_exc_param)
    assert circuit.parameter_count == expected_circuit.parameter_count
    assert circuit._circuit.gates == expected_circuit._circuit.gates
    params = [0.1 * (i + 1) for i in range(circuit.parameter_count)]
    bound_circuit = circuit.bind_parameters(params)
    expected_bound_circuit = expected_circuit.bind_parameters(params)
    assert bound_circuit == expected_bound_circuit


def test_all_singles_doubles_sz() -> None:
    # 3 spin-up electrons (sz = 1.5) in 8 spin orbitals
    circuit = AllSinglesDoubles(8, 3, sz=1.5)
    s_exc, d_exc = excitations(8, 3, occupied_indices=[0, 2, 4])
    assert s_exc == [(0, 6), (2, 6), (4, 6)]
    assert circuit.parameter_count == len(s_exc) + len(d_exc)

    with pytest.warns(FutureWarning, match="sz"):
        AllSinglesDoubles(4, 2)
    with pytest.raises(ValueError):
        AllSinglesDoubles(8, 3, sz=0)
