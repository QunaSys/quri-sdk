from quri_parts.qsub.op import Ident, Op

from . import NS

CNOT: Op = Op(Ident(NS, "CNOT"), 2, self_inverse=True)
