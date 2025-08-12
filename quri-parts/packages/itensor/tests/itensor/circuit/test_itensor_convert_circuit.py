# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import os
from typing import Callable

import juliacall
import numpy
import pytest
from juliacall import Main as jl

from quri_parts.circuit import QuantumCircuit, QuantumGate, gates
from quri_parts.circuit.transpile import (
    IdentityInsertionTranspiler,
    SequentialTranspiler,
)
from quri_parts.itensor.circuit import convert_circuit

abs_dir = os.path.dirname(os.path.abspath(__file__))
library_path = os.path.join(abs_dir, "../../../quri_parts/itensor/library.jl")
include_statement = 'include("' + library_path + '")'
jl.seval(include_statement)
library_path = os.path.join(abs_dir, "circuit_test_library.jl")
jl.seval('include("' + library_path + '")')

jl.seval("using ITensors")
single_qubit_gate_list: list[Callable[[int], QuantumGate]] = [
    gates.Identity,
    gates.X,
    gates.Y,
    gates.Z,
    gates.H,
    gates.S,
    gates.Sdag,
    gates.SqrtX,
    gates.SqrtXdag,
    gates.SqrtY,
    gates.SqrtYdag,
    gates.T,
    gates.Tdag,
]

two_qubit_gate_list: list[Callable[[int, int], QuantumGate]] = [
    gates.CNOT,
    gates.CZ,
    gates.SWAP,
]

three_qubit_gate_list: list[Callable[[int, int, int], QuantumGate]] = [
    gates.TOFFOLI,
]


rotation_gate_list: list[Callable[[int, float], QuantumGate]] = [
    gates.RX,
    gates.RY,
    gates.RZ,
]


def test_convert_single_qubit_gate() -> None:
    qubits = 2
    s: juliacall.VectorValue = jl.siteinds("Qubit", qubits)
    psi: juliacall.AnyValue = jl.init_state(s, qubits)
    expected_list = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0 + 1.0j, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.70710678, 0.70710678, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.5 + 0.5j, 0.5 - 0.5j, 0.0, 0.0],
        [0.5 - 0.5j, 0.5 + 0.5j, 0.0, 0.0],
        [0.5 + 0.5j, 0.5 + 0.5j, 0.0, 0.0],
        [0.5 - 0.5j, -0.5 + 0.5j, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
    ]
    for i, qp_fac in enumerate(single_qubit_gate_list):
        circuit = QuantumCircuit(qubits)
        circuit.add_gate(qp_fac(0))
        psiApplied = jl.apply(convert_circuit(circuit, s), psi)
        stateVector = jl.stateVector(psiApplied, s)
        actual = numpy.array(stateVector)
        assert actual == pytest.approx(expected_list[i], abs=1e-6)


def test_convert_two_qubit_gate() -> None:
    qubits = 2
    s: juliacall.VectorValue = jl.siteinds("Qubit", qubits)
    psi: juliacall.AnyValue = jl.init_state(s, qubits)
    expected_list = [
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ]
    for i, qp_fac in enumerate(two_qubit_gate_list):
        circuit = QuantumCircuit(qubits)
        circuit.add_gate(gates.X(0))
        circuit.add_gate(qp_fac(0, 1))
        psiApplied = jl.apply(convert_circuit(circuit, s), psi)
        stateVector = jl.stateVector(psiApplied, s)
        actual = numpy.array(stateVector)
        assert actual == pytest.approx(expected_list[i], abs=1e-6)


def test_convert_three_qubit_gate() -> None:
    qubits = 3
    s: juliacall.VectorValue = jl.siteinds("Qubit", qubits)
    psi: juliacall.AnyValue = jl.init_state(s, qubits)
    expected_list = [
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    ]
    for i, qp_fac in enumerate(three_qubit_gate_list):
        circuit = QuantumCircuit(qubits)
        circuit.add_gate(gates.X(0))
        circuit.add_gate(gates.X(1))
        circuit.add_gate(qp_fac(0, 1, 2))
        psiApplied = jl.apply(convert_circuit(circuit, s), psi)
        stateVector = jl.stateVector(psiApplied, s)
        actual = numpy.array(stateVector)
        assert actual == pytest.approx(expected_list[i], abs=1e-6)


def test_convert_rotation_gate() -> None:
    qubits = 2
    s: juliacall.VectorValue = jl.siteinds("Qubit", qubits)
    psi: juliacall.AnyValue = jl.init_state(s, qubits)
    expected_list = [
        [0.92387953, 0.0 - 0.38268343j, 0.0, 0.0],
        [0.92387953 + 0.0j, 0.38268343 + 0.0j, 0.0, 0.0],
        [0.92387953 - 0.38268343j, 0.0, 0.0, 0.0],
    ]
    for i, qp_fac in enumerate(rotation_gate_list):
        circuit = QuantumCircuit(qubits)
        circuit.add_gate(qp_fac(0, math.pi / 4))
        psiApplied = jl.apply(convert_circuit(circuit, s), psi)
        stateVector = jl.stateVector(psiApplied, s)
        actual = numpy.array(stateVector)
        assert actual == pytest.approx(expected_list[i], abs=1e-6)


def test_convert_u_gate() -> None:
    qubits = 2
    s: juliacall.VectorValue = jl.siteinds("Qubit", qubits)
    psi: juliacall.AnyValue = jl.init_state(s, qubits)
    circuit = QuantumCircuit(qubits)
    circuit.add_gate(gates.U1(0, 0.5))
    psiApplied = jl.apply(convert_circuit(circuit, s), psi)
    stateVector = jl.stateVector(psiApplied, s)
    actual = numpy.array(stateVector)
    assert actual == pytest.approx(
        [1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j], abs=1e-6
    )

    circuit = QuantumCircuit(qubits)
    circuit.add_gate(gates.U2(0, 0.5, 0.5))
    psiApplied = jl.apply(convert_circuit(circuit, s), psi)
    stateVector = jl.stateVector(psiApplied, s)
    actual = numpy.array(stateVector)
    assert actual == pytest.approx(
        [0.70710678 + 0.0j, 0.62054458 + 0.33900505j, 0.0, 0.0], abs=1e-6
    )

    circuit = QuantumCircuit(qubits)
    circuit.add_gate(gates.U3(0, 0.5, 0.5, 0.5))
    psiApplied = jl.apply(convert_circuit(circuit, s), psi)
    stateVector = jl.stateVector(psiApplied, s)
    actual = numpy.array(stateVector)
    assert actual == pytest.approx(
        [0.96891242 + 0.0j, 0.2171174 + 0.11861178j, 0.0, 0.0], abs=1e-6
    )


def test_convert_pauli_gate() -> None:
    qubits = 2
    s: juliacall.VectorValue = jl.siteinds("Qubit", qubits)
    psi: juliacall.AnyValue = jl.init_state(s, qubits)
    circuit = QuantumCircuit(qubits)
    circuit.add_gate(gates.Pauli((0, 1), (1, 2)))
    psiApplied = jl.apply(convert_circuit(circuit, s), psi)
    stateVector = jl.stateVector(psiApplied, s)
    actual = numpy.array(stateVector)
    assert actual == pytest.approx(
        [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 1.0j], abs=1e-6
    )

    circuit = QuantumCircuit(qubits)
    circuit.add_gate(gates.Pauli((0, 1), (2, 2)))
    with pytest.raises(ValueError):
        convert_circuit(
            circuit, s, SequentialTranspiler([IdentityInsertionTranspiler()])
        )


def test_convert_pauli_rotation_gate() -> None:
    qubits = 2
    s: juliacall.VectorValue = jl.siteinds("Qubit", qubits)
    psi: juliacall.AnyValue = jl.init_state(s, qubits)
    circuit = QuantumCircuit(qubits)
    circuit.add_gate(gates.PauliRotation((0, 1), (1, 3), math.pi / 2))
    psiApplied = jl.apply(convert_circuit(circuit, s), psi)
    stateVector = jl.stateVector(psiApplied, s)
    actual = numpy.array(stateVector)
    assert actual == pytest.approx(
        [
            0.70710678 + 0.0j,
            0.0 - 0.70710678j,
            0.0 + 0.0j,
            0.0 + 0.0j,
        ],
        abs=1e-6,
    )

    circuit = QuantumCircuit(qubits)
    circuit.add_gate(gates.PauliRotation((0, 1), (2, 2), math.pi / 2))
    with pytest.raises(ValueError):
        convert_circuit(
            circuit, s, SequentialTranspiler([IdentityInsertionTranspiler()])
        )
