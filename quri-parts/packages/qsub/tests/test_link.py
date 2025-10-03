from quri_parts.qsub.expand import full_expand
from quri_parts.qsub.lib.std import S, T, Z
from quri_parts.qsub.link import Linker
from quri_parts.qsub.machineinst import MachineSub, Primitive, SubCall
from quri_parts.qsub.qubit import Qubit


def test_link() -> None:
    calltable = {
        Z: MachineSub(
            (Qubit(0),),
            (),
            (),
            (),
            ((SubCall(S, None), (Qubit(0),), ()), (SubCall(S, None), (Qubit(0),), ())),
        ),
        S: MachineSub(
            (Qubit(0),),
            (),
            (),
            (),
            ((Primitive(T), (Qubit(0),), ()), (Primitive(T), (Qubit(0),), ())),
        ),
    }
    sub = MachineSub(
        (Qubit(0),),
        (),
        (),
        (),
        (
            (SubCall(Z, None), (Qubit(0),), ()),
            (SubCall(S, None), (Qubit(0),), ()),
            (Primitive(T), (Qubit(0),), ()),
        ),
    )

    linker = Linker(calltable)
    linked_sub = linker.link(sub)
    expect_linked = MachineSub(
        (Qubit(0),),
        (),
        (),
        (),
        (
            (
                SubCall(
                    Z,
                    MachineSub(
                        (Qubit(0),),
                        (),
                        (),
                        (),
                        (
                            (
                                SubCall(
                                    S,
                                    MachineSub(
                                        (Qubit(0),),
                                        (),
                                        (),
                                        (),
                                        (
                                            (Primitive(T), (Qubit(0),), ()),
                                            (Primitive(T), (Qubit(0),), ()),
                                        ),
                                    ),
                                ),
                                (Qubit(0),),
                                (),
                            ),
                            (
                                SubCall(
                                    S,
                                    MachineSub(
                                        (Qubit(0),),
                                        (),
                                        (),
                                        (),
                                        (
                                            (Primitive(T), (Qubit(0),), ()),
                                            (Primitive(T), (Qubit(0),), ()),
                                        ),
                                    ),
                                ),
                                (Qubit(0),),
                                (),
                            ),
                        ),
                    ),
                ),
                (Qubit(0),),
                (),
            ),
            (
                SubCall(
                    S,
                    MachineSub(
                        (Qubit(0),),
                        (),
                        (),
                        (),
                        (
                            (Primitive(T), (Qubit(0),), ()),
                            (Primitive(T), (Qubit(0),), ()),
                        ),
                    ),
                ),
                (Qubit(0),),
                (),
            ),
            (Primitive(T), (Qubit(0),), ()),
        ),
    )
    assert linked_sub == expect_linked

    expanded_sub = full_expand(linked_sub)
    expect_expand = MachineSub(
        (Qubit(0),),
        (),
        (),
        (),
        (
            (Primitive(T), (Qubit(0),), ()),
            (Primitive(T), (Qubit(0),), ()),
            (Primitive(T), (Qubit(0),), ()),
            (Primitive(T), (Qubit(0),), ()),
            (Primitive(T), (Qubit(0),), ()),
            (Primitive(T), (Qubit(0),), ()),
            (Primitive(T), (Qubit(0),), ()),
        ),
    )
    assert expanded_sub == expect_expand
