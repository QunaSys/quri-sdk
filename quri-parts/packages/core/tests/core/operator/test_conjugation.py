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

from quri_parts.circuit import (
    CNOT,
    CZ,
    SWAP,
    H,
    Identity,
    Pauli,
    S,
    Sdag,
    SqrtX,
    SqrtXdag,
    SqrtY,
    SqrtYdag,
    T,
    X,
    Y,
    Z,
)
from quri_parts.core.operator import PauliLabel, SinglePauli, pauli_label
from quri_parts.core.operator.conjugation import clifford_gate_conjugation


def test_clifford_gate_conjugation() -> None:
    orig_pauli = pauli_label("X0")
    assert clifford_gate_conjugation(Identity(0), orig_pauli) == (
        pauli_label("X0"),
        1.0,
    )
    assert clifford_gate_conjugation(X(0), orig_pauli) == (pauli_label("X0"), 1.0)
    assert clifford_gate_conjugation(Y(0), orig_pauli) == (pauli_label("X0"), -1.0)
    assert clifford_gate_conjugation(Z(0), orig_pauli) == (pauli_label("X0"), -1.0)
    assert clifford_gate_conjugation(H(0), orig_pauli) == (pauli_label("Z0"), 1.0)
    assert clifford_gate_conjugation(S(0), orig_pauli) == (pauli_label("Y0"), 1.0)
    assert clifford_gate_conjugation(Sdag(0), orig_pauli) == (pauli_label("Y0"), -1.0)
    assert clifford_gate_conjugation(SqrtX(0), orig_pauli) == (pauli_label("X0"), 1.0)
    assert clifford_gate_conjugation(SqrtXdag(0), orig_pauli) == (
        pauli_label("X0"),
        1.0,
    )
    assert clifford_gate_conjugation(SqrtY(0), orig_pauli) == (pauli_label("Z0"), -1.0)
    assert clifford_gate_conjugation(SqrtYdag(0), orig_pauli) == (
        pauli_label("Z0"),
        1.0,
    )
    assert clifford_gate_conjugation(CNOT(0, 1), orig_pauli) == (
        pauli_label("X0 X1"),
        1.0,
    )
    assert clifford_gate_conjugation(CNOT(1, 0), orig_pauli) == (pauli_label("X0"), 1.0)
    assert clifford_gate_conjugation(CZ(0, 1), orig_pauli) == (
        pauli_label("X0 Z1"),
        1.0,
    )
    assert clifford_gate_conjugation(CZ(1, 0), orig_pauli) == (
        pauli_label("X0 Z1"),
        1.0,
    )
    assert clifford_gate_conjugation(SWAP(0, 1), orig_pauli) == (pauli_label("X1"), 1.0)
    assert clifford_gate_conjugation(SWAP(1, 0), orig_pauli) == (pauli_label("X1"), 1.0)

    orig_pauli = pauli_label("Y0")
    assert clifford_gate_conjugation(Identity(0), orig_pauli) == (
        pauli_label("Y0"),
        1.0,
    )
    assert clifford_gate_conjugation(X(0), orig_pauli) == (pauli_label("Y0"), -1.0)
    assert clifford_gate_conjugation(Y(0), orig_pauli) == (pauli_label("Y0"), 1.0)
    assert clifford_gate_conjugation(Z(0), orig_pauli) == (pauli_label("Y0"), -1.0)
    assert clifford_gate_conjugation(H(0), orig_pauli) == (pauli_label("Y0"), -1.0)
    assert clifford_gate_conjugation(S(0), orig_pauli) == (pauli_label("X0"), -1.0)
    assert clifford_gate_conjugation(Sdag(0), orig_pauli) == (pauli_label("X0"), 1.0)
    assert clifford_gate_conjugation(SqrtX(0), orig_pauli) == (pauli_label("Z0"), 1.0)
    assert clifford_gate_conjugation(SqrtXdag(0), orig_pauli) == (
        pauli_label("Z0"),
        -1.0,
    )
    assert clifford_gate_conjugation(SqrtY(0), orig_pauli) == (pauli_label("Y0"), 1.0)
    assert clifford_gate_conjugation(SqrtYdag(0), orig_pauli) == (
        pauli_label("Y0"),
        1.0,
    )
    assert clifford_gate_conjugation(CNOT(0, 1), orig_pauli) == (
        pauli_label("Y0 X1"),
        1.0,
    )
    assert clifford_gate_conjugation(CNOT(1, 0), orig_pauli) == (
        pauli_label("Y0 Z1"),
        1.0,
    )
    assert clifford_gate_conjugation(CZ(0, 1), orig_pauli) == (
        pauli_label("Y0 Z1"),
        1.0,
    )
    assert clifford_gate_conjugation(CZ(1, 0), orig_pauli) == (
        pauli_label("Y0 Z1"),
        1.0,
    )
    assert clifford_gate_conjugation(SWAP(0, 1), orig_pauli) == (pauli_label("Y1"), 1.0)
    assert clifford_gate_conjugation(SWAP(1, 0), orig_pauli) == (pauli_label("Y1"), 1.0)

    orig_pauli = pauli_label("Z0")
    assert clifford_gate_conjugation(Identity(0), orig_pauli) == (
        pauli_label("Z0"),
        1.0,
    )
    assert clifford_gate_conjugation(X(0), orig_pauli) == (pauli_label("Z0"), -1.0)
    assert clifford_gate_conjugation(Y(0), orig_pauli) == (pauli_label("Z0"), -1.0)
    assert clifford_gate_conjugation(Z(0), orig_pauli) == (pauli_label("Z0"), 1.0)
    assert clifford_gate_conjugation(H(0), orig_pauli) == (pauli_label("X0"), 1.0)
    assert clifford_gate_conjugation(S(0), orig_pauli) == (pauli_label("Z0"), 1.0)
    assert clifford_gate_conjugation(Sdag(0), orig_pauli) == (pauli_label("Z0"), 1.0)
    assert clifford_gate_conjugation(SqrtX(0), orig_pauli) == (pauli_label("Y0"), -1.0)
    assert clifford_gate_conjugation(SqrtXdag(0), orig_pauli) == (
        pauli_label("Y0"),
        1.0,
    )
    assert clifford_gate_conjugation(SqrtY(0), orig_pauli) == (pauli_label("X0"), 1.0)
    assert clifford_gate_conjugation(SqrtYdag(0), orig_pauli) == (
        pauli_label("X0"),
        -1.0,
    )
    assert clifford_gate_conjugation(CNOT(0, 1), orig_pauli) == (pauli_label("Z0"), 1.0)
    assert clifford_gate_conjugation(CNOT(1, 0), orig_pauli) == (
        pauli_label("Z0 Z1"),
        1.0,
    )
    assert clifford_gate_conjugation(CZ(0, 1), orig_pauli) == (pauli_label("Z0"), 1.0)
    assert clifford_gate_conjugation(CZ(1, 0), orig_pauli) == (pauli_label("Z0"), 1.0)
    assert clifford_gate_conjugation(SWAP(0, 1), orig_pauli) == (pauli_label("Z1"), 1.0)
    assert clifford_gate_conjugation(SWAP(1, 0), orig_pauli) == (pauli_label("Z1"), 1.0)

    for pauli in [SinglePauli.X, SinglePauli.Y, SinglePauli.Z]:
        orig_pauli = PauliLabel({(2, pauli)})  # X2, Y2, or Z2
        assert clifford_gate_conjugation(Identity(0), orig_pauli) == (orig_pauli, 1.0)
        assert clifford_gate_conjugation(X(0), orig_pauli) == (orig_pauli, 1.0)
        assert clifford_gate_conjugation(Y(0), orig_pauli) == (orig_pauli, 1.0)
        assert clifford_gate_conjugation(Z(0), orig_pauli) == (orig_pauli, 1.0)
        assert clifford_gate_conjugation(H(0), orig_pauli) == (orig_pauli, 1.0)
        assert clifford_gate_conjugation(S(0), orig_pauli) == (orig_pauli, 1.0)
        assert clifford_gate_conjugation(Sdag(0), orig_pauli) == (orig_pauli, 1.0)
        assert clifford_gate_conjugation(SqrtX(0), orig_pauli) == (orig_pauli, 1.0)
        assert clifford_gate_conjugation(SqrtXdag(0), orig_pauli) == (orig_pauli, 1.0)
        assert clifford_gate_conjugation(SqrtY(0), orig_pauli) == (orig_pauli, 1.0)
        assert clifford_gate_conjugation(SqrtYdag(0), orig_pauli) == (orig_pauli, 1.0)
        assert clifford_gate_conjugation(CNOT(0, 1), orig_pauli) == (orig_pauli, 1.0)
        assert clifford_gate_conjugation(CNOT(1, 0), orig_pauli) == (orig_pauli, 1.0)
        assert clifford_gate_conjugation(CZ(0, 1), orig_pauli) == (orig_pauli, 1.0)
        assert clifford_gate_conjugation(CZ(1, 0), orig_pauli) == (orig_pauli, 1.0)
        assert clifford_gate_conjugation(SWAP(0, 1), orig_pauli) == (orig_pauli, 1.0)
        assert clifford_gate_conjugation(SWAP(1, 0), orig_pauli) == (orig_pauli, 1.0)


def test_clifford_gate_conjugation_invalid_gate() -> None:
    orig_pauli = pauli_label("X0")
    with pytest.raises(ValueError):
        clifford_gate_conjugation(T(0), orig_pauli)
    with pytest.raises(NotImplementedError):
        clifford_gate_conjugation(
            Pauli(target_indices=(0, 1), pauli_ids=(1, 2)), orig_pauli
        )
