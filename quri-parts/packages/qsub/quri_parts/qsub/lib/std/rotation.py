# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from quri_parts.qsub.op import ParamUnitaryDef, param_op
from quri_parts.qsub.resolve import default_repository
from quri_parts.qsub.sub import Sub, SubBuilder

from . import NS


class _RX(ParamUnitaryDef[float]):
    ns = NS
    name = "RX"
    qubit_count = 1


RX = param_op(_RX)


class _RY(ParamUnitaryDef[float]):
    ns = NS
    name = "RY"
    qubit_count = 1


RY = param_op(_RY)


class _RZ(ParamUnitaryDef[float]):
    ns = NS
    name = "RZ"
    qubit_count = 1


RZ = param_op(_RZ)


class _Phase(ParamUnitaryDef[float]):
    ns = NS
    name = "Phase"
    qubit_count = 1


Phase = param_op(_Phase)


def _phase_to_rz_sub(phase: float) -> Sub:
    b = SubBuilder(1)
    b.add_op(RZ(phase), b.qubits)
    b.add_phase(phase / 2)
    return b.build()


default_repository().register_sub(Phase, _phase_to_rz_sub)


def _rz_to_phase_sub(phase: float) -> Sub:
    b = SubBuilder(1)
    b.add_op(Phase(phase), b.qubits)
    b.add_phase(-phase / 2)
    return b.build()


default_repository().register_sub(RZ, _rz_to_phase_sub)
