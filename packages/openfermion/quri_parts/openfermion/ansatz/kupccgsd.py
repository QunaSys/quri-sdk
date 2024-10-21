# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Sequence

from quri_parts.chem.utils.excitations import DoubleExcitation, SingleExcitation
from quri_parts.circuit import (
    ImmutableLinearMappedParametricQuantumCircuit,
    LinearMappedParametricQuantumCircuit,
    Parameter,
)
from quri_parts.openfermion.transforms import (
    OpenFermionMappingMethods,
    OpenFermionQubitMapperFactory,
    OpenFermionQubitOperatorMapper,
    jordan_wigner,
)
from quri_parts.openfermion.utils import add_exp_excitation_gates_trotter_decomposition


class KUpCCGSD(ImmutableLinearMappedParametricQuantumCircuit):
    """k-unitary pair coupled-cluster generalized singles and doubles
    (k-UpCCGSD) ansatz. The ansatz consists of the exponentials of generalized
    single excitation and pair double excitation operators decomposed by first-
    order Trotter product formula repeated k times. The term "generalized"
    means excitation operators created regardless of the occupied orbitals.
    Here the double excitations are limited to 'pair' excitation, which means
    two electrons from one spatial orbital to another.

    Ref:
        Lee et al., Generalized Unitary Coupled Cluster Wave functions for Quantum
        Computation, J. Chem. Theory Comput. 15, 311–324 (2019),
        `arXiv:1810.02327 <https://arxiv.org/abs/1810.02327>`_.

        PennyLane's documentations,
        `qml.kUpCCGSD <https://docs.pennylane.ai/en/stable/
        code/api/pennylane.kUpCCGSD.html>`_

    Args:
        n_spin_orbitals: Number of spin orbitals.
        k: Number of repetitions.
        fermion_qubit_mapping: Mapping from :class:`FermionOperator` to
          :class:`Operator`
        trotter_number: Number for first-order Trotter product formula.
        delta_sz: Changes of spin in the excitation.
        singlet_excitation: If ``True``, single excitations that start
            from and end at the same spatial orbitals will share the same
            circuit parameter. For the double excitations, the excitations
            are paired, so the singlet excitation condition is satisfied
            automatically.
    """

    def __init__(
        self,
        n_spin_orbitals: int,
        k: int = 1,
        fermion_qubit_mapping: OpenFermionMappingMethods = jordan_wigner,
        trotter_number: int = 1,
        delta_sz: int = 0,
        singlet_excitation: bool = False,
    ):
        mapping = (
            fermion_qubit_mapping(n_spin_orbitals)
            if isinstance(fermion_qubit_mapping, OpenFermionQubitMapperFactory)
            else fermion_qubit_mapping
        )

        assert mapping.n_spin_orbitals == n_spin_orbitals, "n_spin_orbital specified "
        "in the mapping is not consistent with that specified to the first arguement."

        op_mapper = mapping.of_operator_mapper
        n_qubits = mapping.n_qubits

        circuit = LinearMappedParametricQuantumCircuit(n_qubits)

        _construct_circuit(
            circuit=circuit,
            n_spin_orb=n_spin_orbitals,
            delta_sz=delta_sz,
            k=k,
            singlet_excitation=singlet_excitation,
            trotter_number=trotter_number,
            op_mapper=op_mapper,
        )

        super().__init__(circuit)


def _construct_circuit(
    circuit: LinearMappedParametricQuantumCircuit,
    n_spin_orb: int,
    delta_sz: int,
    k: int,
    singlet_excitation: bool,
    trotter_number: int,
    op_mapper: OpenFermionQubitOperatorMapper,
) -> None:
    (
        independent_s_exc,
        independent_s_exc_name,
    ) = _generalized_single_excitations(n_spin_orb, delta_sz, singlet_excitation, k)

    (
        independent_d_exc,
        independent_d_exc_name,
    ) = _generalized_pair_double_excitations(n_spin_orb, k)

    for x in range(k):
        if singlet_excitation:
            spin_symmetric_names_recorder: dict[str, Parameter] = {}
            independent_s_exc_params = []

            for s_exc_name in independent_s_exc_name[x]:
                if s_exc_name in spin_symmetric_names_recorder:
                    param = spin_symmetric_names_recorder[s_exc_name]
                else:
                    param = circuit.add_parameter(s_exc_name)
                    spin_symmetric_names_recorder[s_exc_name] = param
                independent_s_exc_params.append(param)

        else:
            independent_s_exc_params = [
                circuit.add_parameter(name) for name in independent_s_exc_name[x]
            ]
        independent_d_exc_params = [
            circuit.add_parameter(name) for name in independent_d_exc_name[x]
        ]

        for _ in range(trotter_number):
            add_exp_excitation_gates_trotter_decomposition(
                circuit,
                independent_s_exc,
                independent_s_exc_params,
                op_mapper,
                1 / trotter_number,
            )
            add_exp_excitation_gates_trotter_decomposition(
                circuit,
                independent_d_exc,
                independent_d_exc_params,
                op_mapper,
                1 / trotter_number,
            )


def _generalized_single_excitations(
    n_spin_orbitals: int,
    delta_sz: int,
    singlet_excitation: bool,
    k: int,
) -> tuple[Sequence[SingleExcitation], list[list[str]]]:
    """Identify the independent single excitation amplitudes in the KUpGCCSD
    ansatz and their corresponding circuit parameter names."""
    if singlet_excitation:
        assert delta_sz == 0, "delta_sz can only be 0 when singlet_excitation is True."

    sz = [0.5 - (i % 2) for i in range(n_spin_orbitals)]
    s_excitations = []
    for r in range(n_spin_orbitals):
        for p in range(n_spin_orbitals):
            if sz[p] - sz[r] == delta_sz and p != r:
                s_excitations.append((r, p))

    independent_s_exc = []
    independent_s_exc_name: list[list[str]] = [[] for _ in range(k)]

    for s_exc in s_excitations:
        p, q = s_exc
        if (q, p) in independent_s_exc:
            # if p^ q is in s_excs, q^ p is also in s_exc.
            continue
        independent_s_exc.append(s_exc)
        for x in range(k):
            if singlet_excitation:
                independent_s_exc_name[x].append(f"spatialt1_{x}_{p//2}_{q//2}")
            else:
                independent_s_exc_name[x].append(f"t1_{x}_{p}_{q}")

    return independent_s_exc, independent_s_exc_name


def _generalized_pair_double_excitations(
    n_spin_orbitals: int, k: int
) -> tuple[Sequence[DoubleExcitation], list[list[str]]]:
    """Identify the independent paired excitation amplitudes in the KUpGCCSD
    ansatz and their corresponding circuit parameter names."""
    double_excitations = [
        (r, r + 1, p, p + 1)
        for r in range(0, n_spin_orbitals - 1, 2)
        for p in range(0, n_spin_orbitals - 1, 2)
        if p != r
    ]

    independent_d_exc = []
    independent_d_exc_name: list[list[str]] = [[] for _ in range(k)]

    for d_exc in double_excitations:
        p, q, r, s = d_exc
        if (r, s, p, q) in independent_d_exc:
            # if q^ p^ s r is in d_excs, s^ r^ q p is also in d_excs.
            continue
        independent_d_exc.append(d_exc)
        for x in range(k):
            independent_d_exc_name[x].append(f"t2_{x}_{p}_{q}_{r}_{s}")

    return independent_d_exc, independent_d_exc_name
