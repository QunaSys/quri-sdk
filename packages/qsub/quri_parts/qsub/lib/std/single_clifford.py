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
