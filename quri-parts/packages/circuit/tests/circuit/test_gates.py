from collections.abc import Mapping
from typing import Callable

import numpy as np
import pytest

from quri_parts.circuit import (
    CNOT,
    CZ,
    RX,
    RY,
    RZ,
    SWAP,
    TOFFOLI,
    U1,
    U2,
    U3,
    H,
    Identity,
    Measurement,
    ParametricPauliRotation,
    ParametricQuantumGate,
    ParametricRX,
    ParametricRY,
    ParametricRZ,
    Pauli,
    PauliRotation,
    QuantumCircuit,
    QuantumGate,
    S,
    Sdag,
    SingleQubitUnitaryMatrix,
    SqrtX,
    SqrtXdag,
    SqrtY,
    SqrtYdag,
    T,
    Tdag,
    TwoQubitUnitaryMatrix,
    UnitaryMatrix,
    X,
    Y,
    Z,
    gate_names,
)

_factory_name_map: Mapping[Callable[[int], QuantumGate], str] = {
    Identity: gate_names.Identity,
    X: gate_names.X,
    Y: gate_names.Y,
    Z: gate_names.Z,
    H: gate_names.H,
    S: gate_names.S,
    Sdag: gate_names.Sdag,
    SqrtX: gate_names.SqrtX,
    SqrtXdag: gate_names.SqrtXdag,
    SqrtY: gate_names.SqrtY,
    SqrtYdag: gate_names.SqrtYdag,
    T: gate_names.T,
    Tdag: gate_names.Tdag,
}

_rotation_factory_name_map: Mapping[Callable[[int, float], QuantumGate], str] = {
    RX: gate_names.RX,
    RY: gate_names.RY,
    RZ: gate_names.RZ,
}

_parametric_factory_name_map: Mapping[Callable[[int], ParametricQuantumGate], str] = {
    ParametricRX: gate_names.ParametricRX,
    ParametricRY: gate_names.ParametricRY,
    ParametricRZ: gate_names.ParametricRZ,
}


def test_gate_creation() -> None:
    # Single qubit gate
    for f, name in _factory_name_map.items():
        assert f(5) == QuantumGate(name, target_indices=(5,))

    # Single qubit gate with single parameter
    theta = np.random.rand()
    for rf, name in _rotation_factory_name_map.items():
        assert rf(5, theta) == QuantumGate(name, target_indices=(5,), params=(theta,))

    # U gates
    theta, phi, lmd = np.random.rand(3)
    assert U1(5, lmd) == QuantumGate(gate_names.U1, target_indices=(5,), params=(lmd,))
    assert U2(5, phi, lmd) == QuantumGate(
        gate_names.U2, target_indices=(5,), params=(phi, lmd)
    )
    assert U3(5, theta, phi, lmd) == QuantumGate(
        gate_names.U3, target_indices=(5,), params=(theta, phi, lmd)
    )

    # 2 qubit gates
    assert CNOT(5, 7) == QuantumGate(
        gate_names.CNOT, control_indices=(5,), target_indices=(7,)
    )
    assert CZ(5, 7) == QuantumGate(
        gate_names.CZ, control_indices=(5,), target_indices=(7,)
    )
    assert SWAP(5, 7) == QuantumGate(gate_names.SWAP, target_indices=(5, 7))

    # 3 qubit gates
    assert TOFFOLI(2, 5, 7) == QuantumGate(
        gate_names.TOFFOLI, control_indices=(2, 5), target_indices=(7,)
    )

    # Unitary matrix gates
    single_umat = ((1, 0), (0, 1))
    two_umat = ((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1))
    assert SingleQubitUnitaryMatrix(5, single_umat) == QuantumGate(
        gate_names.UnitaryMatrix,
        target_indices=(5,),
        unitary_matrix=single_umat,
    )
    assert TwoQubitUnitaryMatrix(5, 7, two_umat) == QuantumGate(
        gate_names.UnitaryMatrix,
        target_indices=(5, 7),
        unitary_matrix=two_umat,
    )
    assert UnitaryMatrix((7, 11), two_umat) == QuantumGate(
        gate_names.UnitaryMatrix,
        target_indices=(7, 11),
        unitary_matrix=two_umat,
    )
    with pytest.raises(ValueError):
        SingleQubitUnitaryMatrix(5, two_umat)  # Wrong shape
    with pytest.raises(ValueError):
        TwoQubitUnitaryMatrix(5, 7, single_umat)  # Wrong shape
    with pytest.raises(ValueError):
        UnitaryMatrix((5, 7, 11), ((0,) * 8,) * 8)  # Non unitary

    # Pauli gates
    target_indices = (1, 3, 5)
    pauli_ids = (3, 1, 2)
    theta = np.random.rand()
    assert Pauli(target_indices, pauli_ids) == QuantumGate(
        gate_names.Pauli, target_indices=target_indices, pauli_ids=pauli_ids
    )
    assert PauliRotation(target_indices, pauli_ids, theta) == QuantumGate(
        gate_names.PauliRotation,
        target_indices=target_indices,
        pauli_ids=pauli_ids,
        params=(theta,),
    )

    # Parametric gates
    for pf, name in _parametric_factory_name_map.items():
        assert pf(5) == ParametricQuantumGate(name, target_indices=(5,))
    assert ParametricPauliRotation(target_indices, pauli_ids) == ParametricQuantumGate(
        gate_names.ParametricPauliRotation,
        target_indices=target_indices,
        pauli_ids=pauli_ids,
    )

    # Measurement gate
    assert Measurement([5], [7]) == QuantumGate(
        gate_names.Measurement, target_indices=(5,), classical_indices=(7,)
    )


def test_gate_addition() -> None:
    theta, phi, lmd = np.random.rand(3)
    target_indices = (2, 0, 1)
    pauli_ids = (3, 1, 2)
    single_umat = ((1, 0), (0, 1))
    two_umat = ((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1))

    lc = QuantumCircuit(3, cbit_count=1)
    gates = [
        Identity(0),
        X(0),
        Y(0),
        Z(0),
        H(0),
        S(0),
        Sdag(0),
        SqrtX(0),
        SqrtXdag(0),
        SqrtY(0),
        SqrtYdag(0),
        T(0),
        Tdag(0),
        RX(0, theta),
        RY(0, theta),
        RZ(0, theta),
        U1(0, lmd),
        U2(0, phi, lmd),
        U3(0, theta, phi, lmd),
        CNOT(0, 1),
        CZ(0, 1),
        SWAP(0, 1),
        TOFFOLI(0, 1, 2),
        SingleQubitUnitaryMatrix(0, single_umat),
        TwoQubitUnitaryMatrix(0, 1, two_umat),
        UnitaryMatrix((0, 1), two_umat),
        Pauli(target_indices, pauli_ids),
        PauliRotation(target_indices, pauli_ids, theta),
        Measurement([0], [0]),
    ]
    lc.extend(gates)

    mc = QuantumCircuit(3, cbit_count=1)
    mc.add_Identity_gate(0)
    mc.add_X_gate(0)
    mc.add_Y_gate(0)
    mc.add_Z_gate(0)
    mc.add_H_gate(0)
    mc.add_S_gate(0)
    mc.add_Sdag_gate(0)
    mc.add_SqrtX_gate(0)
    mc.add_SqrtXdag_gate(0)
    mc.add_SqrtY_gate(0)
    mc.add_SqrtYdag_gate(0)
    mc.add_T_gate(0)
    mc.add_Tdag_gate(0)
    mc.add_RX_gate(0, theta)
    mc.add_RY_gate(0, theta)
    mc.add_RZ_gate(0, theta)
    mc.add_U1_gate(0, lmd)
    mc.add_U2_gate(0, phi, lmd)
    mc.add_U3_gate(0, theta, phi, lmd)
    mc.add_CNOT_gate(0, 1)
    mc.add_CZ_gate(0, 1)
    mc.add_SWAP_gate(0, 1)
    mc.add_TOFFOLI_gate(0, 1, 2)
    mc.add_SingleQubitUnitaryMatrix_gate(0, single_umat)
    mc.add_TwoQubitUnitaryMatrix_gate(0, 1, two_umat)
    mc.add_UnitaryMatrix_gate((0, 1), two_umat)
    mc.add_Pauli_gate(target_indices, pauli_ids)
    mc.add_PauliRotation_gate(target_indices, pauli_ids, theta)
    mc.measure(0, 0)

    assert lc == mc


def test_gate_equal() -> None:
    # Quantum Gate
    assert X(0) == X(0)
    assert not X(0) == X(1)
    assert not X(0) == H(1)
    assert H(1) == H(1)
    assert RX(0, 0.1) == RX(0, 0.1)
    assert not RX(0, 0.1) == RX(0, 0.2)
    assert U2(1, 0.1, 0.2) == U2(1, 0.1, 0.2)
    assert not U2(1, 0.1, 0.2) == U2(1, 0.2, 0.2)
    assert not U2(1, 0.1, 0.2) == U2(1, 0.1, 0.1)
    assert U3(2, 0.1, 0.2, 0.3) == U3(2, 0.1, 0.2, 0.3)
    assert not U3(2, 0.1, 0.2, 0.3) == U3(2, 0.2, 0.2, 0.3)
    assert not U3(2, 0.1, 0.2, 0.3) == U3(2, 0.1, 0.1, 0.3)
    assert not U3(2, 0.1, 0.2, 0.3) == U3(2, 0.1, 0.2, 0.2)
    assert CNOT(0, 1) == CNOT(0, 1)
    assert not CNOT(0, 1) == CNOT(1, 0)
    assert TOFFOLI(0, 1, 2) == TOFFOLI(0, 1, 2)
    assert TOFFOLI(0, 1, 2) == TOFFOLI(1, 0, 2)
    assert not TOFFOLI(0, 1, 2) == TOFFOLI(0, 2, 1)
    assert UnitaryMatrix(
        (0, 1), ((1, 0, 0, 0), (0, 0, 0, 1), (0, 0, 1, 0), (0, 1, 0, 0))
    ) == UnitaryMatrix((0, 1), ((1, 0, 0, 0), (0, 0, 0, 1), (0, 0, 1, 0), (0, 1, 0, 0)))
    assert not UnitaryMatrix(
        (0, 1), ((1, 0, 0, 0), (0, 0, 0, 1), (0, 0, 1, 0), (0, 1, 0, 0))
    ) == UnitaryMatrix((1, 0), ((1, 0, 0, 0), (0, 0, 0, 1), (0, 0, 1, 0), (0, 1, 0, 0)))
    assert not UnitaryMatrix(
        (0, 1), ((1, 0, 0, 0), (0, 0, 0, 1), (0, 0, 1, 0), (0, 1, 0, 0))
    ) == UnitaryMatrix((0, 1), ((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)))
    assert Pauli((0, 1), (1, 3)) == Pauli((0, 1), (1, 3))
    assert not Pauli((0, 1), (1, 3)) == Pauli((1, 0), (1, 3))
    assert Pauli((0, 1), (1, 3)) == Pauli((1, 0), (3, 1))
    assert PauliRotation((1, 2, 3), (3, 2, 1), 0.1) == PauliRotation(
        (1, 2, 3), (3, 2, 1), 0.1
    )
    assert not PauliRotation((1, 2, 3), (3, 2, 1), 0.2) == PauliRotation(
        (1, 2, 3), (3, 2, 1), 0.1
    )
    assert PauliRotation((2, 1, 3), (2, 3, 1), 0.1) == PauliRotation(
        (1, 2, 3), (3, 2, 1), 0.1
    )
    assert not PauliRotation((2, 1, 3), (3, 2, 1), 0.1) == PauliRotation(
        (1, 2, 3), (3, 2, 1), 0.1
    )
    assert Measurement([0, 1], [1, 0]) == Measurement([0, 1], [1, 0])
    assert not Measurement([0, 1], [1, 0]) == Measurement([0, 1], [0, 1])
    assert Measurement([0, 1], [1, 0]) == Measurement([1, 0], [0, 1])

    # ParametricQuantumGate
    assert ParametricRX(0) == ParametricRX(0)
    assert not ParametricRX(0) == ParametricRX(1)
    assert not ParametricRX(0) == ParametricRY(0)
    assert ParametricPauliRotation((0, 1, 2), (2, 1, 3)) == ParametricPauliRotation(
        (0, 1, 2), (2, 1, 3)
    )
    assert not ParametricPauliRotation((0, 1, 2), (2, 1, 3)) == ParametricPauliRotation(
        (0, 1, 2), (1, 2, 3)
    )
    assert ParametricPauliRotation((0, 1, 2), (2, 1, 3)) == ParametricPauliRotation(
        (0, 2, 1), (2, 3, 1)
    )
