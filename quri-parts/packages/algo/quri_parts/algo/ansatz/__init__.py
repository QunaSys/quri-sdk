# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from .hardware_efficient import HardwareEfficient, HardwareEfficientReal
from .symmetry_preserving import SymmetryPreserving, SymmetryPreservingReal
from .two_local import (
    EntanglementPatternType,
    EntLayerMaker,
    EntLayerMakerArg,
    RotLayerMaker,
    RotLayerMakerArg,
    TwoLocal,
    build_entangler_map,
)
from .z2_symmetry_preserving import Z2SymmetryPreservingReal

#: Function to rotate a qubit in a rotation layer.
RotLayerMaker = RotLayerMaker
#: Function to entangle a qubit pair in an entanglement layer.
EntLayerMaker = EntLayerMaker


__all__ = [
    "EntanglementPatternType",
    "RotLayerMaker",
    "EntLayerMaker",
    "RotLayerMakerArg",
    "EntLayerMakerArg",
    "TwoLocal",
    "build_entangler_map",
    "HardwareEfficient",
    "HardwareEfficientReal",
    "SymmetryPreserving",
    "SymmetryPreservingReal",
    "Z2SymmetryPreservingReal",
]
