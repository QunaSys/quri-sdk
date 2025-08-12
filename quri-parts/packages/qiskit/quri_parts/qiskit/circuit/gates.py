# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Literal

from quri_parts.circuit import QuantumGate
from quri_parts.qiskit.circuit import gate_names


class ECRFactory:
    name: Literal["ECR"] = gate_names.ECR

    def __call__(self, target_index1: int, target_index2: int) -> QuantumGate:
        return QuantumGate(
            name=self.name,
            target_indices=(target_index1, target_index2),
        )


ECR = ECRFactory()
r"""IBM Quantum native gate ECR defined as follows.

.. math::
    ECR = \frac{1}{\sqrt{2}}
    \begin{pmatrix}
        0 & 1 & 0 & i \\
        1 & 0 & -i & 0 \\
        0 & i & 0 & 1 \\
        -i & 0 & 1 & 0
    \end{pmatrix}

Ref:
    https://docs.quantum.ibm.com/api/qiskit/qiskit.circuit.library.ECRGate
"""
