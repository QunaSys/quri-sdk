from quri_parts.qsub.expand import full_expand
from quri_parts.qsub.link import Linker
from quri_parts.qsub.machineinst import MachineSub, Primitive, SubCall
from quri_parts.qsub.namespace import NameSpace
from quri_parts.qsub.op import Ident, Op
from quri_parts.qsub.qubit import Qubit

NS = NameSpace("test")
F = Op(Ident(NS, "F"), 3, 0)
G = Op(Ident(NS, "G"), 3, 0)
H = Op(Ident(NS, "H"), 2, 0)
K = Op(Ident(NS, "K"), 2, 0)


def test_expand() -> None:
    calltable = {
        F: MachineSub(
            (Qubit(0), Qubit(1), Qubit(2)),
            (),
            (),
            (),
            (
                (Primitive(G), (Qubit(0), Qubit(1), Qubit(2)), ()),
                (SubCall(H, None), (Qubit(0), Qubit(1)), ()),
            ),
        ),
        H: MachineSub(
            (Qubit(0), Qubit(1)),
            (),
            (Qubit(2),),
            (),
            ((Primitive(K), (Qubit(1), Qubit(2)), ()),),
        ),
    }
    sub = MachineSub(
        (Qubit(0), Qubit(1), Qubit(2)),
        (),
        (),
        (),
        ((SubCall(F, None), (Qubit(0), Qubit(1), Qubit(2)), ()),),
    )

    linker = Linker(calltable)
    linked_sub = linker.link(sub)

    expanded_sub = full_expand(linked_sub)
    expect_expand = MachineSub(
        (Qubit(0), Qubit(1), Qubit(2)),
        (),
        (Qubit(3),),
        (),
        (
            (Primitive(G), (Qubit(0), Qubit(1), Qubit(2)), ()),
            (Primitive(K), (Qubit(1), Qubit(3)), ()),
        ),
    )
    assert expanded_sub == expect_expand
