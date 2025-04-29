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
