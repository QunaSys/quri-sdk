# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from quri_parts.qsub.op import Ident, Op

from . import NS

Identity = Op(Ident(NS, "Identity"), 1, self_inverse=True)
H = Op(Ident(NS, "H"), 1, self_inverse=True)
X = Op(Ident(NS, "X"), 1, self_inverse=True)
Y = Op(Ident(NS, "Y"), 1, self_inverse=True)
Z = Op(Ident(NS, "Z"), 1, self_inverse=True)
S = Op(Ident(NS, "S"), 1)
Sdag = Op(Ident(NS, "Sdag"), 1)
SqrtX = Op(Ident(NS, "SqrtX"), 1)
SqrtXdag = Op(Ident(NS, "SqrtXdag"), 1)
SqrtY = Op(Ident(NS, "SqrtY"), 1)
SqrtYdag = Op(Ident(NS, "SqrtYdag"), 1)
