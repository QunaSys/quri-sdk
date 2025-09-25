from dataclasses import dataclass
from typing import Literal, Optional, Sequence

from quri_parts.chem.mol import ActiveSpace
from typing_extensions import TypeAlias

from quri_algo.problem.operators.hamiltonian import (
    FermionicHamiltonian,
    QubitHamiltonian,
)

from .interface import HamiltonianMixin

AtomCoordinate: TypeAlias = tuple[float, float, float]


@dataclass
class MolecularSystem(HamiltonianMixin):
    atom: Sequence[tuple[str, AtomCoordinate]] | str
    basis: str = "sto-3g"
    spin: int = 0
    charge: int = 0
    active_space: Optional[ActiveSpace] = None
    backend: Literal[
        "pyscf_mem_efficient", "pyscf_density_fitting"
    ] = "pyscf_mem_efficient"

    def get_qubit_hamiltonian(self) -> QubitHamiltonian:
        raise NotImplementedError("Not supported")

    def get_fermionic_hamiltonian(self) -> FermionicHamiltonian:
        raise NotImplementedError("Not supported")
