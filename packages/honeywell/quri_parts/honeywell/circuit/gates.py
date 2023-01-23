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
from quri_parts.honeywell.circuit import gate_names


class U1qFactory:
    name: Literal["U1q"] = gate_names.U1q

    def __call__(self, target_index: int, theta: float, phi: float) -> QuantumGate:
        return QuantumGate(
            name=self.name,
            target_indices=(target_index,),
            params=(theta, phi),
        )


U1q = U1qFactory()
r"""Honeywell native gate U1q defined as follows.

.. math::
    U_{1q}(\theta, \phi) =
    e^{-i(\cos{\phi \hat{X}} + \sin{\phi \hat{Y}}) \theta / 2} =
    \begin{pmatrix}
        \cos{\frac{\theta}{2}} & -ie^{-i\phi} \sin{\frac{\theta}{2}} \\
        -ie^{i\phi} \sin{\frac{\theta}{2}} & \cos{\frac{\theta}{2}}
    \end{pmatrix}

Ref:
    [1]: https://www.quantinuum.com/hardware/h1
        System Model H1 Product Data Sheet (P4 Native Gate Set)
"""


class ZZFactory:
    name: Literal["ZZ"] = gate_names.ZZ

    def __call__(self, target_index0: int, target_index1: int) -> QuantumGate:
        return QuantumGate(
            name=self.name,
            target_indices=(target_index0, target_index1),
        )


ZZ = ZZFactory()
r"""Honeywell native gate ZZ defined as follows.

.. math::
    ZZ = e^{-i\frac{\pi}{4} \hat{Z} \otimes \hat{Z}} = e^{-\frac{i\pi}{4}}
    \begin{pmatrix}
        1 & 0 & 0 & 0 \\
        0 & i & 0 & 0 \\
        0 & 0 & i & 0 \\
        0 & 0 & 0 & 1
    \end{pmatrix}

Ref:
    [1]: https://www.quantinuum.com/hardware/h1
        System Model H1 Product Data Sheet (P4 Native Gate Set)
"""


class RZZFactory:
    name: Literal["RZZ"] = gate_names.RZZ

    def __call__(
        self, target_index0: int, target_index1: int, theta: float
    ) -> QuantumGate:
        return QuantumGate(
            name=self.name,
            target_indices=(target_index0, target_index1),
            params=(theta,),
        )


RZZ = RZZFactory()
r"""Honeywell native gate RZZ defined as follows.

.. math::
    RZZ = e^{-i\frac{\pi}{2} \hat{Z} \otimes \hat{Z}} = e^{-\frac{i\pi}{2}}
    \begin{pmatrix}
        1 & 0 & 0 & 0 \\
        0 & e^{i\theta} & 0 & 0 \\
        0 & 0 & e^{i\theta} & 0 \\
        0 & 0 & 0 & 1
    \end{pmatrix}

Ref:
    [1]: https://www.quantinuum.com/hardware/h1
        System Model H1 Product Data Sheet (P4 Native Gate Set)
"""
