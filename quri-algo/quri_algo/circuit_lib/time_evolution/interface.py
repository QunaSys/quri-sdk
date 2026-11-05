# Licensed under the MIT License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      https://mit-license.org/
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# NOTE: This module has a deprecated counterpart at
# quri_algo/qsub/time_evolution/interface.py which re-exports from here.
from typing import Any, Protocol

from quri_parts.qsub.op import OpFactory

from quri_algo.problem import QubitHamiltonian


class TimeEvolutionOpFactory(OpFactory[QubitHamiltonian, float, Any], Protocol):
    ...
