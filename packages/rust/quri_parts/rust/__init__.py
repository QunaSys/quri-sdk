import sys
from typing import Any

from quri_parts.rust.quri_parts_rust import (  # type: ignore[import-untyped]
    circuit,
    qulacs,
)

# circuit
sys.modules["quri_parts.rust.circuit"] = circuit
sys.modules["quri_parts.rust.circuit.circuit"] = circuit.circuit
sys.modules["quri_parts.rust.circuit.gate"] = circuit.gate
sys.modules["quri_parts.rust.circuit.gates"] = circuit.gates
sys.modules["quri_parts.rust.circuit.circuit"] = circuit.circuit
sys.modules["quri_parts.rust.circuit.parameter"] = circuit.parameter
sys.modules["quri_parts.rust.circuit.circuit_parametric"] = circuit.circuit_parametric
sys.modules["quri_parts.rust.circuit.noise"] = circuit.noise

# qulacs
sys.modules["quri_parts.rust.qulacs"] = qulacs

_old__iadd__ = circuit.circuit_parametric.ParametricQuantumCircuit.__iadd__


def _new__iadd__(
    lhs: circuit.circuit_parametric.ParametricQuantumCircuit,
    rhs: Any,
) -> Any:
    try:
        return _old__iadd__(lhs, rhs)
    except NotImplementedError:
        return NotImplemented


setattr(circuit.circuit_parametric.ParametricQuantumCircuit, "__iadd__", _new__iadd__)

__all__ = [
    "circuit",
    "qulacs",
]
