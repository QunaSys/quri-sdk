from quri_parts.qsub.op import Ident, Op

from . import NS

SWAP = Op(Ident(NS, "SWAP"), 2, self_inverse=True)
