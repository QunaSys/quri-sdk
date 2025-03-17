from quri_parts.qsub.op import Ident, Op

from . import NS

Toffoli = Op(Ident(NS, "Toffoli"), 3, self_inverse=True)
