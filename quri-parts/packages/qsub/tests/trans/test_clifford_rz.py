# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import Counter

from quri_parts.qsub.compile import compile_sub
from quri_parts.qsub.eval import QURIPartsEvaluatorHooks
from quri_parts.qsub.evaluate import Evaluator
from quri_parts.qsub.lib.std import CNOT, RX, RY, RZ, H, Toffoli
from quri_parts.qsub.primitive import CliffordRZ
from quri_parts.qsub.sub import SubBuilder
from quri_parts.qsub.trans.clifford_rz import CliffordRZSetTranspiler

_ALLOWED = {
    "Identity",
    "H",
    "X",
    "Y",
    "Z",
    "SqrtX",
    "SqrtXdag",
    "SqrtY",
    "SqrtYdag",
    "S",
    "Sdag",
    "CNOT",
    "CZ",
    "SWAP",
    "RZ",
}


def test_clifford_rz_set_transpiler() -> None:
    b = SubBuilder(2)
    q0, q1 = b.qubits
    b.add_op(H, (q0,))
    b.add_op(RX(0.5), (q0,))
    b.add_op(RY(0.7), (q1,))
    b.add_op(CNOT, (q0, q1))
    b.add_op(RZ(0.3), (q1,))
    sub = b.build()

    transpiled = CliffordRZSetTranspiler()(sub)
    compiled = compile_sub(transpiled, CliffordRZ, sub_transpilers=())
    circuit = Evaluator(QURIPartsEvaluatorHooks()).run(compiled)
    names = [gate.name for gate in circuit.gates]
    assert set(names) <= _ALLOWED
    # RX/RY are rewritten away; the native RZ rotation survives.
    assert "RX" not in names and "RY" not in names and "RZ" in names


def test_clifford_rz_set_transpiler_toffoli() -> None:
    b = SubBuilder(3)
    q0, q1, q2 = b.qubits
    b.add_op(H, (q0,))
    b.add_op(Toffoli, (q0, q1, q2))
    b.add_op(RZ(0.3), (q2,))
    sub = b.build()

    transpiled = CliffordRZSetTranspiler()(sub)
    compiled = compile_sub(transpiled, CliffordRZ, sub_transpilers=())
    circuit = Evaluator(QURIPartsEvaluatorHooks()).run(compiled)
    names = [gate.name for gate in circuit.gates]
    assert set(names) <= _ALLOWED
    assert Counter(names) == Counter({"H": 3, "CNOT": 6, "RZ": 8})
