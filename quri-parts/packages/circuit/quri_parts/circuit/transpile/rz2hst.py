from typing import Callable, Optional

import mpmath  # type: ignore[import-untyped]
import numpy as np
from numpy.typing import NDArray

from quri_parts.circuit import ImmutableQuantumCircuit, QuantumCircuit, gate_names
from quri_parts.circuit.transpile.transpiler import CircuitTranspilerProtocol

#: 2x2 unitaries of the gates gridsynth emits, used to recover the global phase.
_GATE_MATRICES: dict[str, NDArray[np.complex128]] = {
    "H": np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2),
    "S": np.array([[1, 0], [0, 1j]], dtype=np.complex128),
    "T": np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=np.complex128),
    "X": np.array([[0, 1], [1, 0]], dtype=np.complex128),
}


def _rz_matrix(theta: float) -> NDArray[np.complex128]:
    return np.array(
        [[np.exp(-0.5j * theta), 0], [0, np.exp(0.5j * theta)]], dtype=np.complex128
    )


def call_gridsynth(theta: float, epsilon: float) -> str:
    """Synthesize an ``RZ(theta)`` rotation into a gridsynth gate-sequence
    string over ``H``, ``S``, ``T``, ``X``, up to a global phase.

    This is the default decomposition backend of :class:`RZ2HSTTranspiler` and
    requires the optional ``pygridsynth`` package.

    Args:
        theta: Rotation angle (radians) of the ``RZ`` gate to approximate.
        epsilon: Target approximation precision passed to gridsynth.

    Returns:
        A string of gate symbols (each one of ``H``, ``S``, ``T``, ``X``) whose
        product approximates ``RZ(theta)`` up to a global phase, to within
        ``epsilon``.
    """
    import inspect

    from pygridsynth.gridsynth import gridsynth_gates

    # We always synthesize up to a global phase (up_to_phase=True). With
    # up_to_phase=False gridsynth would also have to reproduce the exact global
    # phase using W tokens, and since the required phase is generally not a
    # multiple of pi/4 (the only phase a W token can supply) it must be
    # approximated by a much longer H/S/T sequence. Synthesizing only up to a
    # phase keeps the sequence short; the exact global phase is recovered
    # separately (see RZ2HSTTranspiler.__call__).
    #
    # pygridsynth >= 2 accepts ``up_to_phase`` (via **kwargs); pygridsynth 1.x
    # (the last Python 3.9-compatible line) has no such parameter but already
    # synthesizes up to a global phase by default. Pass it only when supported.
    params = inspect.signature(gridsynth_gates).parameters
    supports_up_to_phase = "up_to_phase" in params or any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()
    )
    kwargs = {"up_to_phase": True} if supports_up_to_phase else {}
    result: str = gridsynth_gates(mpmath.mpf(theta), mpmath.mpf(epsilon), **kwargs)
    return result


def _add_gridsynth_gate(circuit: QuantumCircuit, qubit: int, symbol: str) -> None:
    if symbol == "H":
        circuit.add_H_gate(qubit)
    elif symbol == "S":
        circuit.add_S_gate(qubit)
    elif symbol == "T":
        circuit.add_T_gate(qubit)
    elif symbol == "X":
        circuit.add_X_gate(qubit)
    else:
        raise ValueError(f"Unsupported gridsynth symbol: {symbol!r}")


class RZ2HSTTranspiler(CircuitTranspilerProtocol):
    """A transpiler that replaces each RZ gate with a gridsynth-generated
    sequence of H, S, T, and X gates.

    Non-RZ gates are kept unchanged. The decomposition precision is controlled
    by ``epsilon`` and passed to the gridsynth function.

    Args:
        epsilon: Precision of the decomposition. Defaults to 1.0e-5.
        gridsynth: An optional callable with signature
            ``(theta: float, epsilon: float) -> str`` that returns a gate
            sequence string (e.g. ``"HSTX"``) over ``H``, ``S``, ``T``, ``X``.
            When *None* (the default),
            :func:`pygridsynth.gridsynth.gridsynth_gates` is used.
    """

    def __init__(
        self,
        epsilon: float = 1.0e-5,
        gridsynth: Optional[Callable[[float, float], str]] = None,
    ):
        self._epsilon = epsilon
        self._gridsynth = gridsynth if gridsynth is not None else call_gridsynth
        #: Global phase (radians) accumulated in the most recent call, such that
        #: ``exp(i * phase)`` times the emitted circuit equals the input RZ gates.
        self.phase: float = 0.0

    def __call__(self, circuit: ImmutableQuantumCircuit) -> ImmutableQuantumCircuit:
        self.phase = 0.0
        result = QuantumCircuit(circuit.qubit_count, circuit.cbit_count)
        for gate in circuit.gates:
            if gate.name == gate_names.RZ:
                qubit = gate.target_indices[0]
                theta = gate.params[0]
                unitary: NDArray[np.complex128] = np.eye(2, dtype=np.complex128)
                for symbol in self._gridsynth(theta, self._epsilon):
                    # pygridsynth 1.x emits a global-phase 'W' token; it carries
                    # no gate and is subsumed by the phase recovered below.
                    if symbol == "W":
                        continue
                    _add_gridsynth_gate(result, qubit, symbol)
                    unitary = _GATE_MATRICES[symbol] @ unitary
                # gridsynth matched RZ(theta) only up to a global phase, so
                # recover that phase by aligning the synthesized unitary with the
                # exact RZ(theta) matrix (exp(i * phase) * unitary == RZ(theta)).
                self.phase += float(np.angle(np.vdot(unitary, _rz_matrix(theta))))
            else:
                result.add_gate(gate)
        return result
