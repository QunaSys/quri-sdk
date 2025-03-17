from quri_parts.qsub.op import Ident, Op

from . import NS

CZ: Op = Op(Ident(NS, "CZ"), 2, self_inverse=True)
