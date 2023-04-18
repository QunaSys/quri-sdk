# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np

from quri_parts.circuit.gate import QuantumGate
from quri_parts.circuit.gate_names import CLIFFORD_GATE_NAMES


def is_clifford(gate: QuantumGate, rtol: float = 1.0e-5, atol: float = 1.0e-8) -> bool:
    r"""Returns True if the input gate is Clifford, otherwise False. In the case of a
    rotation gate, whether it is a Clifford gate or not is determined by whether the
    rotation angle is in the Clifford angle set :math:`\{\pi n /2| n\in\mathbb{Z}\}`
    or not.
    This can be applied only to the gates in the set CLIFFORD_GATE_NAMES and currently
    implemented rotation gates: {RX, RY, RZ, U1, U2, U3, PauliRotation}. Since newly
    defined gates (e.g. U'(a, b) = RZ(2*a)RY(a+2b)) may return the wrong result, in
    this case this function currently returns a NotImplementedError.

    Args:
        gate: A gate to be checked whether Clifford or not.
        rtol: The relative tolerance parameter determines how close the angle of
            rotation is to the angle in the Clifford angle set.
        atol: The absolute tolerance parameter determines how close the angle of
            rotation is to the angle in the Clifford angle set.
    """
    if gate.name in CLIFFORD_GATE_NAMES:
        return True

    elif gate.name in {"T", "Tdag"}:
        return False

    elif gate.name in {"RX", "RY", "RZ", "U1", "U2", "U3", "PauliRotation"}:
        angle_list = [2 * param / np.pi for param in gate.params]
        if all([np.isclose(round(angle), angle, rtol, atol) for angle in angle_list]):
            return True
        else:
            return False

    else:
        raise NotImplementedError(
            f"A code to determine whether input gate\
            {gate.name} is Clifford gate or not has not been implemented."
        )
