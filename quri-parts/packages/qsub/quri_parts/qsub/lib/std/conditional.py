# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from contextlib import contextmanager
from typing import Generator

from quri_parts.qsub.op import Ident, Op
from quri_parts.qsub.register import Register
from quri_parts.qsub.sub import SubBuilder

from . import NS

# Compare and branch on zero.
Cbz = Op(Ident(NS, "Cbz"), 0, 2, unitary=False)
Label = Op(Ident(NS, "Label"), 0, 1, unitary=False)


@contextmanager
def conditional(builder: SubBuilder, r0: Register) -> Generator[None, None, None]:
    l0 = builder.add_aux_register()
    # When the value of Cbz's first register is 0,
    # branch to first Label with Cbz's second register.
    builder.add_op(Cbz, (), (r0, l0))
    yield
    builder.add_op(Label, (), (l0,))
