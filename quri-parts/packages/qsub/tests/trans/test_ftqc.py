# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from math import pi

import pytest

from quri_parts.qsub.compile import compile_sub
from quri_parts.qsub.eval import QURIPartsEvaluatorHooks
from quri_parts.qsub.evaluate import Evaluator
from quri_parts.qsub.lib.std import CNOT, RZ, H
from quri_parts.qsub.primitive import FTQCBasicSet
from quri_parts.qsub.sub import SubBuilder
from quri_parts.qsub.trans.ftqc import FTQCSetTranspiler

_ALLOWED = {"H", "S", "T", "CNOT"}


def test_ftqc_set_transpiler_named_angles() -> None:
    b = SubBuilder(2)
    q0, q1 = b.qubits
    b.add_op(H, (q0,))
    b.add_op(RZ(pi / 4), (q0,))
    b.add_op(RZ(-pi / 2), (q1,))
    b.add_op(CNOT, (q0, q1))
    b.add_op(RZ(3 * pi / 4), (q0,))
    sub = b.build()

    transpiled = FTQCSetTranspiler()(sub)
    compiled = compile_sub(transpiled, FTQCBasicSet, sub_transpilers=())
    circuit = Evaluator(QURIPartsEvaluatorHooks()).run(compiled)
    assert {gate.name for gate in circuit.gates} <= _ALLOWED


@pytest.mark.gridsynth
def test_ftqc_set_transpiler_arbitrary_angle() -> None:
    b = SubBuilder(2)
    q0, q1 = b.qubits
    b.add_op(H, (q0,))
    b.add_op(RZ(0.3), (q0,))
    b.add_op(CNOT, (q0, q1))
    sub = b.build()

    # A non-pi/4 angle is approximated into {H, S, T, CNOT} via gridsynth.
    transpiled = FTQCSetTranspiler()(sub)
    compiled = compile_sub(transpiled, FTQCBasicSet, sub_transpilers=())
    circuit = Evaluator(QURIPartsEvaluatorHooks()).run(compiled)
    names = {gate.name for gate in circuit.gates}
    assert names <= _ALLOWED
    assert "T" in names
