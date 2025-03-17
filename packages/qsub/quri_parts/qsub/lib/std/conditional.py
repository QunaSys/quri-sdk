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
