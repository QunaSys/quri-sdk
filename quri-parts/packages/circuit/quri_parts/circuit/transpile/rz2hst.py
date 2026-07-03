import os
import subprocess
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


#: A gridsynth driver maps ``(theta, epsilon)`` to ``(gates, phase)``: a
#: gate-sequence string over ``H``, ``S``, ``T``, ``X`` approximating
#: ``RZ(theta)`` up to a global phase, and that global phase in radians (or
#: ``nan`` when the driver does not provide it, in which case
#: :class:`RZ2HSTTranspiler` recovers it from the gate matrices).
GridsynthDriver = Callable[[float, float], "tuple[str, float]"]

_GRIDSYNTH_ENV_KEY = "GRIDSYNTH_PATH"


def driver_pygridsynth(up_to_phase: bool = True) -> GridsynthDriver:
    """Build the default gridsynth driver, backed by the pure-Python
    ``pygridsynth`` package.

    The returned callable maps ``(theta, epsilon)`` to a gate-sequence string
    over ``H``, ``S``, ``T``, ``X`` approximating ``RZ(theta)`` up to a global
    phase, to within ``epsilon`` (the global phase is recovered separately by
    :meth:`RZ2HSTTranspiler.__call__`).
    """

    def driver(theta: float, epsilon: float) -> "tuple[str, float]":
        import inspect

        from pygridsynth.gridsynth import gridsynth_gates

        # We always synthesize up to a global phase (up_to_phase=True). With
        # up_to_phase=False gridsynth would also have to reproduce the exact
        # global phase using W tokens, and since the required phase is generally
        # not a multiple of pi/4 (the only phase a W token can supply) it must be
        # approximated by a much longer H/S/T sequence. Synthesizing only up to a
        # phase keeps the sequence short.
        #
        # pygridsynth >= 2 accepts ``up_to_phase`` (via **kwargs); pygridsynth 1.x
        # (the last Python 3.9-compatible line) has no such parameter but already
        # synthesizes up to a global phase by default. Pass it only when supported.
        params = inspect.signature(gridsynth_gates).parameters
        supports_up_to_phase = "up_to_phase" in params or any(
            p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()
        )
        kwargs = {"up_to_phase": up_to_phase} if supports_up_to_phase else {}
        result: str = gridsynth_gates(mpmath.mpf(theta), mpmath.mpf(epsilon), **kwargs)
        # gridsynth_gates does not report the global phase; signal "not
        # provided" so the transpiler recovers it from the gate matrices.
        return result, float("nan")

    return driver


def driver_cli(up_to_phase: bool = True) -> GridsynthDriver:
    """Build a gridsynth driver that shells out to the external ``gridsynth``
    command-line tool (an alternative to :func:`driver_pygridsynth`).

    The executable is taken from the ``GRIDSYNTH_PATH`` environment variable
    when set, otherwise the ``gridsynth`` command on ``PATH`` is used. It is
    invoked with ``-p`` so the decomposition is only up to a global phase (the
    phase is recovered separately by :meth:`RZ2HSTTranspiler.__call__`).
    """

    def driver(theta: float, epsilon: float) -> "tuple[str, float]":
        executable = os.environ.get(_GRIDSYNTH_ENV_KEY, "gridsynth")
        command = [executable, "-e", str(epsilon)]
        if up_to_phase:
            command.append("-p")
        command += ["--", str(theta)]
        try:
            proc = subprocess.run(command, check=False, capture_output=True, text=True)
        except FileNotFoundError as e:
            raise RuntimeError(f"gridsynth command was not found: {executable}") from e
        if proc.returncode != 0:
            stderr = proc.stderr.strip()
            raise RuntimeError(
                f"gridsynth failed with code {proc.returncode}: "
                f"{stderr or '<no stderr>'}"
            )
        word = proc.stdout.strip()
        if not word:
            raise RuntimeError("gridsynth returned an empty decomposition.")
        # The CLI does not report the global phase; signal "not provided".
        return word, float("nan")

    return driver


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
        gridsynth: An optional :data:`GridsynthDriver` with signature
            ``(theta: float, epsilon: float) -> tuple[str, float]`` returning a
            gate-sequence string (e.g. ``"HSTX"``) over ``H``, ``S``, ``T``,
            ``X`` and the global phase in radians (``nan`` if not provided, in
            which case the phase is recovered from the gate matrices). When
            *None* (the default), :func:`driver_pygridsynth` is used;
            :func:`driver_cli` is an alternative backed by the external
            ``gridsynth`` command.
    """

    def __init__(
        self,
        epsilon: float = 1.0e-5,
        gridsynth: Optional[GridsynthDriver] = None,
    ):
        self._epsilon = epsilon
        self._gridsynth = gridsynth if gridsynth is not None else driver_pygridsynth()
        self.phase: float = 0.0

    def __call__(self, circuit: ImmutableQuantumCircuit) -> ImmutableQuantumCircuit:
        self.phase = 0.0
        result = QuantumCircuit(circuit.qubit_count, circuit.cbit_count)
        for gate in circuit.gates:
            if gate.name == gate_names.RZ:
                qubit = gate.target_indices[0]
                theta = gate.params[0]
                gates, phase = self._gridsynth(theta, self._epsilon)
                unitary: NDArray[np.complex128] = np.eye(2, dtype=np.complex128)
                for symbol in gates:
                    if symbol == "W":
                        continue
                    _add_gridsynth_gate(result, qubit, symbol)
                    unitary = _GATE_MATRICES[symbol] @ unitary
                if np.isnan(phase):
                    # The driver did not report the global phase; recover it by
                    # aligning the synthesized unitary with the exact RZ(theta)
                    # matrix (exp(i * phase) * unitary == RZ(theta)).
                    phase = float(np.angle(np.vdot(unitary, _rz_matrix(theta))))
                self.phase += phase
            else:
                result.add_gate(gate)
        return result
