# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from ._namespace import NS
from .cnot import CNOT
from .conditional import Cbz, Label, conditional
from .control import Controlled
from .cz import CZ
from .logic import scoped_and, scoped_and_clifford_t, scoped_and_single_toffoli
from .measure import M
from .multi_control import MultiControlled
from .multi_control_gates import (
    MCH,
    MCRX,
    MCRY,
    MCRZ,
    MCS,
    MCT,
    MCX,
    MCY,
    MCZ,
    MCPhase,
    MCSdag,
    MCSqrtX,
    MCSqrtXdag,
    MCSqrtY,
    MCSqrtYdag,
    MCTdag,
)
from .multipauli import Pauli, PauliRotation
from .rotation import RX, RY, RZ, Phase
from .single_clifford import (
    H,
    Identity,
    S,
    Sdag,
    SqrtX,
    SqrtXdag,
    SqrtY,
    SqrtYdag,
    X,
    Y,
    Z,
)
from .swap import SWAP
from .t import T, Tdag
from .toffoli import Toffoli

# should be imported after multi_control
from .inverse import Inverse  # isort: skip

__all__ = [
    "Cbz",
    "CNOT",
    "CZ",
    "H",
    "Identity",
    "Inverse",
    "Label",
    "M",
    "MCH",
    "MCRX",
    "MCRY",
    "MCRZ",
    "MCS",
    "MCSdag",
    "MCSqrtX",
    "MCSqrtXdag",
    "MCSqrtY",
    "MCSqrtYdag",
    "MCT",
    "MCTdag",
    "MCPhase",
    "MCX",
    "MCY",
    "MCZ",
    "Phase",
    "NS",
    "Pauli",
    "PauliRotation",
    "Phase",
    "RX",
    "RY",
    "RZ",
    "S",
    "Sdag",
    "SqrtX",
    "SqrtXdag",
    "SqrtY",
    "SqrtYdag",
    "SWAP",
    "T",
    "Tdag",
    "Toffoli",
    "X",
    "Y",
    "Z",
    "conditional",
    "Controlled",
    "MultiControlled",
    "scoped_and",
    "scoped_and_clifford_t",
    "scoped_and_single_toffoli",
]
