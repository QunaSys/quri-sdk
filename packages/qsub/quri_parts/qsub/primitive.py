from quri_parts.qsub.lib.std import (
    CNOT,
    CZ,
    RX,
    RY,
    RZ,
    SWAP,
    H,
    Identity,
    Phase,
    S,
    Sdag,
    SqrtX,
    SqrtXdag,
    SqrtY,
    SqrtYdag,
    T,
    Tdag,
    Toffoli,
    X,
    Y,
    Z,
)

SingleQubitClifford = (Identity, H, X, Y, Z, SqrtX, SqrtXdag, SqrtY, SqrtYdag, S, Sdag)
TwoQubitClifford = (CNOT, CZ, SWAP)
Clifford = SingleQubitClifford + TwoQubitClifford
Rotation = (RX, RY, RZ, Phase)
CliffordT = Clifford + (T, Tdag)
CliffordRZ = Clifford + (RZ,)

RotationSet = Rotation + TwoQubitClifford
RZSet = (X, SqrtX, RZ, CNOT)
FTQCBasicSet = (H, S, T, CNOT)
AllBasicSet = CliffordT + Rotation + (Toffoli,)
