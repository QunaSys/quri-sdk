# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
