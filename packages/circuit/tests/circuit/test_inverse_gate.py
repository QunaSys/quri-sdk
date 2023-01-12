from typing import Callable

import numpy as np

from quri_parts.circuit import (
    RX,
    RY,
    RZ,
    U1,
    U2,
    U3,
    QuantumGate,
    S,
    Sdag,
    SqrtX,
    SqrtXdag,
    SqrtY,
    SqrtYdag,
    T,
    Tdag,
    inverse_gate,
)

_inverse_pairs: list[
    tuple[Callable[[int], QuantumGate], Callable[[int], QuantumGate]]
] = [(S, Sdag), (SqrtX, SqrtXdag), (SqrtY, SqrtYdag), (T, Tdag)]
_single_param_rotations: list[Callable[[int, float], QuantumGate]] = [RX, RY, RZ, U1]


def _assert_inverse_gates(a: QuantumGate, b: QuantumGate) -> None:
    assert a == inverse_gate(b)
    assert inverse_gate(a) == b


def test_inverse_gate() -> None:
    for lf, rf in _inverse_pairs:
        _assert_inverse_gates(lf(7), rf(7))

    for f in _single_param_rotations:
        theta = np.random.rand()
        _assert_inverse_gates(f(7, theta), f(7, -theta))

    theta, phi, lmd = np.random.rand(3)
    _assert_inverse_gates(U2(7, phi, lmd), U2(7, -phi, -lmd))
    _assert_inverse_gates(U3(7, theta, phi, lmd), U3(7, -theta, -phi, -lmd))
