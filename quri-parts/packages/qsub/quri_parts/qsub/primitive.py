# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from quri_parts.qsub.lib.std import (
    CNOT,
    CZ,
    RX,
    RY,
    RZ,
    SWAP,
    H,
    Identity,
    Phase,
    S,
    Sdag,
    SqrtX,
    SqrtXdag,
    SqrtY,
    SqrtYdag,
    T,
    Tdag,
    Toffoli,
    X,
    Y,
    Z,
)

SingleQubitClifford = (Identity, H, X, Y, Z, SqrtX, SqrtXdag, SqrtY, SqrtYdag, S, Sdag)
TwoQubitClifford = (CNOT, CZ, SWAP)
Clifford = SingleQubitClifford + TwoQubitClifford
Rotation = (RX, RY, RZ, Phase)
CliffordT = Clifford + (T, Tdag)
CliffordRZ = Clifford + (RZ,)

RotationSet = Rotation + TwoQubitClifford
RZSet = (X, SqrtX, RZ, CNOT)
FTQCBasicSet = (H, S, T, CNOT)
AllBasicSet = CliffordT + Rotation + (Toffoli,)
