import math

import pytest

from quri_parts.qsub.lib.std import CNOT, RZ, SWAP, Phase
from quri_parts.qsub.resolve import default_repository, resolve_sub


def test_phase_to_rz_conversion() -> None:
    phase = math.pi / 7
    sub = resolve_sub(Phase(phase), default_repository())
    assert sub is not None
    assert tuple(sub.operations) == ((RZ(phase), sub.qubits, ()),)
    assert sub.phase == pytest.approx(phase / 2)


def test_rz_to_phase_conversion() -> None:
    phase = -math.pi / 11
    sub = resolve_sub(RZ(phase), default_repository())
    assert sub is not None
    assert tuple(sub.operations) == ((Phase(phase), sub.qubits, ()),)
    assert sub.phase == pytest.approx(-phase / 2)


def test_swap_to_cnot_conversion() -> None:
    sub = resolve_sub(SWAP, default_repository())
    assert sub is not None
    q0, q1 = sub.qubits
    assert tuple(sub.operations) == (
        (CNOT, (q0, q1), ()),
        (CNOT, (q1, q0), ()),
        (CNOT, (q0, q1), ()),
    )
    assert sub.phase == 0
