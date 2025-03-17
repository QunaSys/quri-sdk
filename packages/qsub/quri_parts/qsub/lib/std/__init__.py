from ._namespace import NS
from .cnot import CNOT
from .conditional import Cbz, Label, conditional
from .control import Controlled, MultiControlled
from .cz import CZ
from .inverse import Inverse
from .logic import scoped_and, scoped_and_clifford_t, scoped_and_single_toffoli
from .measure import M
from .multipauli import Pauli, PauliRotation
from .rotation import RX, RY, RZ, Phase
from .single_clifford import (
    H,
    Identity,
    S,
    Sdag,
    SqrtX,
    SqrtXdag,
    SqrtY,
    SqrtYdag,
    X,
    Y,
    Z,
)
from .swap import SWAP
from .t import T, Tdag
from .toffoli import Toffoli

__all__ = [
    "Cbz",
    "CNOT",
    "CZ",
    "H",
    "Identity",
    "Inverse",
    "Label",
    "M",
    "NS",
    "Pauli",
    "PauliRotation",
    "Phase",
    "RX",
    "RY",
    "RZ",
    "S",
    "Sdag",
    "SqrtX",
    "SqrtXdag",
    "SqrtY",
    "SqrtYdag",
    "SWAP",
    "T",
    "Tdag",
    "Toffoli",
    "X",
    "Y",
    "Z",
    "conditional",
    "Controlled",
    "MultiControlled",
    "scoped_and",
    "scoped_and_clifford_t",
    "scoped_and_single_toffoli",
]
