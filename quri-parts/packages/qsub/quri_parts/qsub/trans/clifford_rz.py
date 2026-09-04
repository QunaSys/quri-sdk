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
    CliffordRZSetTranspiler as _CircuitCliffordRZSetTranspiler,
)
from quri_parts.qsub.sub import Sub

from .qp_trans import SeparateQURIPartsTranspiler
from .transpiler import SubTranspilerProtocol


class CliffordRZSetTranspiler(SubTranspilerProtocol):
    """A qsub Sub transpiler that transpiles a ``Sub`` into the Clifford+RZ
    gate set (Clifford gates plus the continuous RZ rotation).

    It wraps :class:`quri_parts.circuit.transpile.CliffordRZSetTranspiler` with
    :class:`~quri_parts.qsub.trans.qp_trans.SeparateQURIPartsTranspiler` so that
    it operates on a ``Sub`` rather than a ``QuantumCircuit``.

    ``epsilon`` (default ``1e-9``) is the tolerance for snapping a near-Clifford
    RZ angle to a named gate; it is not a gridsynth synthesis precision (RZ stays
    native, so no angle approximation is performed). Only the top-level
    operations of the ``Sub`` are rewritten; gates inside nested subroutines are
    left untouched until those subroutines are expanded.
    """

    def __init__(self, epsilon: float = 1.0e-9) -> None:
        self._transpiler = SeparateQURIPartsTranspiler(
            [_CircuitCliffordRZSetTranspiler(epsilon)]
        )

    def __call__(self, sub: Sub) -> Sub:
        return self._transpiler(sub)
