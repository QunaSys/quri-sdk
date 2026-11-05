# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from quri_parts.circuit.transpile import (
    CliffordTSetTranspiler as _CircuitCliffordTSetTranspiler,
)
from quri_parts.qsub.sub import Sub

from .qp_trans import SeparateQURIPartsTranspiler
from .transpiler import SubTranspilerProtocol


class CliffordTSetTranspiler(SubTranspilerProtocol):
    """A qsub Sub transpiler that transpiles a ``Sub`` into the Clifford+T gate
    set.

    ``epsilon`` (default ``1e-9``) is both the tolerance for matching a
    fused RZ angle to a named gate and the gridsynth synthesis precision
    for the approximated rotations.
    """

    def __init__(self, epsilon: float = 1.0e-9) -> None:
        self._transpiler = SeparateQURIPartsTranspiler(
            [_CircuitCliffordTSetTranspiler(epsilon)]
        )

    def __call__(self, sub: Sub) -> Sub:
        return self._transpiler(sub)
