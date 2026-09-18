# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest

from quri_parts.qsub.compile import compile_sub
from quri_parts.qsub.eval import QURIPartsEvaluatorHooks
from quri_parts.qsub.evaluate import Evaluator
from quri_parts.qsub.lib import std
from quri_parts.qsub.lib.std import CNOT, RZ, H
from quri_parts.qsub.sub import SubBuilder
from quri_parts.qsub.trans.rz2hst import RZ2HSTTranspiler

_ALLOWED = {"H", "S", "T", "X", "CNOT"}


@pytest.mark.gridsynth
def test_rz2hst_transpiler() -> None:
    b = SubBuilder(2)
    q0, q1 = b.qubits
    b.add_op(H, (q0,))
    b.add_op(RZ(0.2), (q0,))
    b.add_op(CNOT, (q0, q1))
    b.add_op(RZ(0.5), (q1,))
    sub = b.build()

    transpiled = RZ2HSTTranspiler(epsilon=1.0e-3)(sub)
    compiled = compile_sub(transpiled, (std.H, std.S, std.T, std.X, std.CNOT))
    circuit = Evaluator(QURIPartsEvaluatorHooks()).run(compiled)
    assert {gate.name for gate in circuit.gates} <= _ALLOWED
