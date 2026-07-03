# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional

from quri_parts.circuit.transpile.rz2hst import GridsynthDriver
from quri_parts.circuit.transpile.rz2hst import (
    RZ2HSTTranspiler as _CircuitRZ2HSTTranspiler,
)
from quri_parts.qsub.sub import Sub

from .qp_trans import SeparateQURIPartsTranspiler
from .transpiler import SubTranspilerProtocol


class RZ2HSTTranspiler(SubTranspilerProtocol):
    """A qsub Sub transpiler that decomposes every RZ rotation in a ``Sub``
    into a sequence of H, S, T (and X) gates using gridsynth, to a target
    precision ``epsilon`` (default ``1e-5``, the gridsynth approximation
    error).

    Args:
        epsilon: Precision of the decomposition.
        gridsynth: An optional :data:`GridsynthDriver` passed through to the
            wrapped circuit-level ``RZ2HSTTranspiler``.
    """

    def __init__(
        self,
        epsilon: float = 1.0e-5,
        gridsynth: Optional[GridsynthDriver] = None,
    ) -> None:
        self._transpiler = SeparateQURIPartsTranspiler(
            [_CircuitRZ2HSTTranspiler(epsilon, gridsynth)]
        )

    def __call__(self, sub: Sub) -> Sub:
        return self._transpiler(sub)
