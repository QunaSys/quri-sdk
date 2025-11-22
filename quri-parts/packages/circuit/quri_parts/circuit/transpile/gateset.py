# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections.abc import Iterable, Mapping, Sequence
from math import pi
from typing import cast

from quri_parts.circuit import (
    ImmutableQuantumCircuit,
    LinearMappedParametricQuantumCircuit,
    LinearParameterMapping,
    ParametricQuantumCircuitProtocol,
    QuantumCircuit,
)
from quri_parts.circuit import gates as gf
from quri_parts.circuit.gate import QuantumGate
from quri_parts.circuit.gate_names import (
    CLIFFORD_GATE_NAMES,
    CNOT,
    CZ,
    RX,
    RY,
    RZ,
    SINGLE_QUBIT_GATE_NAMES,
    SWAP,
    TOFFOLI,
    U1,
    U2,
    U3,
    CliffordGateNameType,
    GateNameType,
    H,
    Identity,
    ParametricPauliRotation,
    ParametricRX,
    ParametricRY,
    ParametricRZ,
    Pauli,
    PauliRotation,
    S,
    Sdag,
    SqrtX,
    SqrtXdag,
    SqrtY,
    SqrtYdag,
    T,
    Tdag,
    UnitaryMatrix,
    X,
    Y,
    Z,
)

from .fuse import (
    FuseRotationTranspiler,
    NormalizeRotationTranspiler,
    Rotation2NamedTranspiler,
    ZeroRotationEliminationTranspiler,
)
from .gate_kind_decomposer import (
    CNOT2CZHTranspiler,
    CZ2CNOTHTranspiler,
    H2RXRYTranspiler,
    RX2RZSqrtXTranspiler,
    RY2RZSqrtXTranspiler,
    S2RZTranspiler,
    Sdag2RZTranspiler,
    SqrtX2RXTranspiler,
    SqrtXdag2RXTranspiler,
    SqrtY2RYTranspiler,
    SqrtYdag2RYTranspiler,
    SWAP2CNOTTranspiler,
    T2RZTranspiler,
    Tdag2RZTranspiler,
    TOFFOLI2HTTdagCNOTTranspiler,
    U1ToRZTranspiler,
    U2ToRXRZTranspiler,
    U3ToRXRZTranspiler,
    X2RXTranspiler,
    Y2RYTranspiler,
    Z2RZTranspiler,
)
from .identity_manipulation import IdentityEliminationTranspiler
from .multi_pauli_decomposer import (
    PauliDecomposeTranspiler,
    PauliRotationDecomposeTranspiler,
)
from .transpiler import (
    CircuitTranspiler,
    CircuitTranspilerProtocol,
    GateKindDecomposer,
    ParametricCircuitTranspilerProtocol,
    SequentialTranspiler,
)
from .unitary_matrix_decomposer import (
    SingleQubitUnitaryMatrix2RYRZTranspiler,
    TwoQubitUnitaryMatrixKAKTranspiler,
)

# Clifford conversion table of patterns with fewer gate species, ordered by gate length.
# Conversions to two gates are covered, otherwise only major patterns are retained.
_equiv_clifford_table: Mapping[str, list[list[str]]] = {
    H: [[X, SqrtYdag], [Z, SqrtY], [SqrtY, X], [SqrtYdag, Z], [S, SqrtX, S]],
    X: [
        [Y, Z],
        [H, SqrtY],
        [SqrtX, SqrtX],
        [SqrtXdag, SqrtXdag],
        [SqrtYdag, H],
        [H, Z, H],
        [H, S, S, H],
    ],
    Y: [
        [Z, X],
        [SqrtY, SqrtY],
        [SqrtYdag, SqrtYdag],
        [S, S, X],
        [Z, H, Z, H],
        [S, S, H, S, S, H],
    ],
    Z: [
        [X, Y],
        [S, S],
        [H, SqrtYdag],
        [SqrtY, H],
        [Sdag, Sdag],
        [H, X, H],
        [X, S, S, X],
    ],
    SqrtX: [[X, SqrtXdag], [Sdag, H, Sdag], [S, Z, H, Z, S], [S, S, S, H, S, S, S]],
    SqrtXdag: [[X, SqrtX], [S, H, S], [Z, SqrtX, Z], [S, S, SqrtX, S, S]],
    SqrtY: [
        [H, X],
        [Z, H],
        [Y, SqrtYdag],
        [S, S, H],
        [Sdag, SqrtX, S],
        [Z, S, SqrtX, S],
        [S, S, S, SqrtX, S],
    ],
    SqrtYdag: [
        [H, Z],
        [X, H],
        [Y, SqrtY],
        [H, S, S],
        [S, SqrtX, Sdag],
        [S, SqrtX, Z, S],
        [S, SqrtX, S, S, S],
    ],
    S: [[Z, Sdag], [H, SqrtX, H], [Sdag, Sdag, Sdag]],
    Sdag: [[Z, S], [SqrtX, H, SqrtX], [S, S, S]],
}


class CliffordConversionTranspiler(CircuitTranspilerProtocol):
    """A CircuitTranspiler that converts Clifford gates in a circuit into the
    desired Clifford gate sequences.

    Convert the Clifford gates in the circuit into gate sequences containing only the
    user-specified Clifford gates. Such conversions are done on a best-effort basis and
    may leave non target Clifford gates in the output.

    Clifford gates that could not be converted or non-Clifford gates will remain in
    place.

    Args:
        target_gateset: A Sequence of Clifford gate names to output.
    """

    def __init__(self, target_gateset: Iterable[CliffordGateNameType]):
        self._gateset = set(target_gateset)
        if self._gateset - (CLIFFORD_GATE_NAMES & SINGLE_QUBIT_GATE_NAMES):
            raise ValueError(
                "Target gateset must contain only single qubit clifford gates."
            )

    def __call__(self, circuit: ImmutableQuantumCircuit) -> ImmutableQuantumCircuit:
        ret = []
        cache: dict[str, list[str]] = {}

        for gate in circuit.gates:
            if gate.name not in CLIFFORD_GATE_NAMES & SINGLE_QUBIT_GATE_NAMES:
                ret.append(gate)
                continue

            if gate.name in self._gateset:
                ret.append(gate)
                continue

            if gate.name not in _equiv_clifford_table:
                ret.append(gate)
                continue

            if gate.name in cache:
                ret.extend(
                    QuantumGate(name=name, target_indices=gate.target_indices)
                    for name in cache[gate.name]
                )
                continue

            for candidate in _equiv_clifford_table[gate.name]:
                if set(candidate) <= self._gateset:
                    cache[gate.name] = candidate
                    ret.extend(
                        QuantumGate(name=name, target_indices=gate.target_indices)
                        for name in candidate
                    )
                    break
            else:
                cache[gate.name] = [gate.name]
                ret.append(gate)

        return QuantumCircuit(circuit.qubit_count, gates=ret)


class RZ2RXRYTranspiler(GateKindDecomposer):
    """A CircuitTranspiler that converts RZ gates in a circuit into the gate
    sequences containing RX and RY gates."""

    @property
    def target_gate_names(self) -> Sequence[str]:
        return [RZ]

    def decompose(self, gate: QuantumGate) -> Sequence[QuantumGate]:
        qubit = gate.target_indices[0]
        theta = gate.params[0]
        return [
            gf.RX(qubit, pi / 2.0),
            gf.RY(qubit, -theta),
            gf.RX(qubit, -pi / 2.0),
        ]


class RY2RXRZTranspiler(GateKindDecomposer):
    """A CircuitTranspiler that converts RY gates in a circuit into the gate
    sequences containing RX and RZ gates."""

    @property
    def target_gate_names(self) -> Sequence[str]:
        return [RY]

    def decompose(self, gate: QuantumGate) -> Sequence[QuantumGate]:
        qubit = gate.target_indices[0]
        theta = gate.params[0]
        return [
            gf.RX(qubit, pi / 2.0),
            gf.RZ(qubit, theta),
            gf.RX(qubit, -pi / 2.0),
        ]


class RX2RYRZTranspiler(GateKindDecomposer):
    """A CircuitTranspiler that converts RX gates in a circuit into the gate
    sequences containing RY and RZ gates."""

    @property
    def target_gate_names(self) -> Sequence[str]:
        return [RX]

    def decompose(self, gate: QuantumGate) -> Sequence[QuantumGate]:
        qubit = gate.target_indices[0]
        theta = gate.params[0]
        return [
            gf.RZ(qubit, pi / 2.0),
            gf.RY(qubit, theta),
            gf.RZ(qubit, -pi / 2.0),
        ]


class RX2RZHTranspiler(GateKindDecomposer):
    """A CircuitTranspiler that converts RX gates in a circuit into the gate
    sequences containing RZ and H gates."""

    @property
    def target_gate_names(self) -> Sequence[str]:
        return [RX]

    def decompose(self, gate: QuantumGate) -> Sequence[QuantumGate]:
        qubit = gate.target_indices[0]
        theta = gate.params[0]
        return [
            gf.H(qubit),
            gf.RZ(qubit, theta),
            gf.H(qubit),
        ]


class RY2RZHTranspiler(GateKindDecomposer):
    """A CircuitTranspiler that converts RY gates in a circuit into the gate
    sequences containing RZ and H gates."""

    @property
    def target_gate_names(self) -> Sequence[str]:
        return [RY]

    def decompose(self, gate: QuantumGate) -> Sequence[QuantumGate]:
        qubit = gate.target_indices[0]
        theta = gate.params[0]
        return [
            gf.RZ(qubit, -pi / 2.0),
            gf.H(qubit),
            gf.RZ(qubit, theta),
            gf.H(qubit),
            gf.RZ(qubit, pi / 2.0),
        ]


class ParametricRX2RZHTranspiler(ParametricCircuitTranspilerProtocol):
    def __call__(
        self, circuit: ParametricQuantumCircuitProtocol
    ) -> LinearMappedParametricQuantumCircuit:
        ret = LinearMappedParametricQuantumCircuit(
            circuit.qubit_count, circuit.cbit_count
        )
        ret._param_mapping = LinearParameterMapping(circuit.param_mapping.in_params)
        pmap = circuit.param_mapping.mapping

        for gate, param in circuit.primitive_circuit().gates_and_params:
            if isinstance(gate, QuantumGate):
                ret.add_gate(gate)
            else:
                if param is None:
                    raise ValueError("Parametric gate with no Parameter: {gate}")

                qubit = gate.target_indices[0]
                if gate.name == ParametricRX:
                    ret.add_H_gate(qubit)
                    ret.add_ParametricRZ_gate(qubit, pmap[param])
                    ret.add_H_gate(qubit)
                elif gate.name == ParametricRY:
                    ret.add_ParametricRY_gate(qubit, pmap[param])
                elif gate.name == ParametricRZ:
                    ret.add_ParametricRZ_gate(qubit, pmap[param])
                elif gate.name == ParametricPauliRotation:
                    ret.add_ParametricPauliRotation_gate(
                        gate.target_indices,
                        gate.pauli_ids,
                        pmap[param],
                    )
                else:
                    raise ValueError(f"Unsupported parametric gate: {gate.name}")

        return ret


class ParametricRY2RZHTranspiler(ParametricCircuitTranspilerProtocol):
    def __call__(
        self, circuit: ParametricQuantumCircuitProtocol
    ) -> LinearMappedParametricQuantumCircuit:
        ret = LinearMappedParametricQuantumCircuit(
            circuit.qubit_count, circuit.cbit_count
        )
        ret._param_mapping = LinearParameterMapping(circuit.param_mapping.in_params)
        pmap = circuit.param_mapping.mapping

        for gate, param in circuit.primitive_circuit().gates_and_params:
            if isinstance(gate, QuantumGate):
                ret.add_gate(gate)
            else:
                if param is None:
                    raise ValueError("Parametric gate with no Parameter: {gate}")

                qubit = gate.target_indices[0]
                if gate.name == ParametricRX:
                    ret.add_ParametricRX_gate(qubit, pmap[param])
                elif gate.name == ParametricRY:
                    qubit = gate.target_indices[0]
                    ret.add_RZ_gate(qubit, -pi / 2.0)
                    ret.add_H_gate(qubit)
                    ret.add_ParametricRZ_gate(qubit, pmap[param])
                    ret.add_H_gate(qubit)
                    ret.add_RZ_gate(qubit, pi / 2.0)
                elif gate.name == ParametricRZ:
                    ret.add_ParametricRZ_gate(qubit, pmap[param])
                elif gate.name == ParametricPauliRotation:
                    ret.add_ParametricPauliRotation_gate(
                        gate.target_indices,
                        gate.pauli_ids,
                        pmap[param],
                    )
                else:
                    raise ValueError(f"Unsupported parametric gate: {gate.name}")

        return ret


class IdentityTranspiler(CircuitTranspilerProtocol):
    """A CircuitTranspiler returns the same circuit as the input."""

    def __call__(self, circuit: ImmutableQuantumCircuit) -> ImmutableQuantumCircuit:
        return circuit


class RotationConversionTranspiler(CircuitTranspilerProtocol):
    """A CircuitTranspiler that converts rotation gates (RX, RY, and RZ) in a
    circuit to each other.

    Convert rotation gates in a circuit into the gate sequences containing only
    specified kinds of rotation gates and Clifford gates.

    If there are 2 permitted rotation gate kinds, the rotation gates are converted
    into the rotation gates only; if there is 1 permitted rotation gate kind, the
    rotation gates are converted into the rotation gates and Clifford gates.

    The more favorable Clifford gates can be indicated. However, the choice of Clifford
    gates is made on a best-effort basis. Clifford gate kinds included in the output
    is not guaranteed.

    Args:
        target_rotation: A Sequence of rotation gate names to output.
        favorable_clifford: A Sequence of Clifford gate names to be prioritesed as an
            output.
    """

    def __init__(
        self,
        target_rotation: Iterable[GateNameType],
        favorable_clifford: Iterable[CliffordGateNameType] = (),
    ):
        self._target_rotation = set(target_rotation)
        self._favorable_clifford = set(favorable_clifford)
        self._decomposer = self._construct_decomposer()

        if self._target_rotation - {RX, RY, RZ}:
            raise ValueError("Unsupported target rotation gate kinds are specified.")
        if self._favorable_clifford - (CLIFFORD_GATE_NAMES & SINGLE_QUBIT_GATE_NAMES):
            raise ValueError(
                "Non single qubit Clifford gates are specified for favorable_clifford."
            )

    def _construct_decomposer(self) -> CircuitTranspiler:
        rot_to_trans_map = {
            frozenset({RX, RY, RZ}): IdentityTranspiler(),
            frozenset({RX, RY}): RZ2RXRYTranspiler(),
            frozenset({RY, RZ}): RX2RYRZTranspiler(),
            frozenset({RX, RZ}): RY2RXRZTranspiler(),
            frozenset({RZ}): SequentialTranspiler(
                [RX2RZHTranspiler(), RY2RZHTranspiler()]
            ),
            # Support {RX}, {RY}, and {} here in the future.
        }

        if H not in self._favorable_clifford and SqrtX in self._favorable_clifford:
            rot_to_trans_map[frozenset({RZ})] = SequentialTranspiler(
                [RX2RZSqrtXTranspiler(), RY2RZSqrtXTranspiler()]
            )
            # Support {RX}, {RY}, and {} here in the future.

        return rot_to_trans_map.get(
            frozenset(self._target_rotation), IdentityTranspiler()
        )

    def _validate(self, circuit: ImmutableQuantumCircuit) -> None:
        for gate in circuit.gates:
            if gate.name in {RX, RY, RZ} and gate.name not in self._target_rotation:
                raise ValueError(f"{gate} cannot be converted into the target gateset.")

    def __call__(self, circuit: ImmutableQuantumCircuit) -> ImmutableQuantumCircuit:
        tr_circuit = self._decomposer(circuit)
        self._validate(tr_circuit)
        return tr_circuit


class GateSetConversionTranspiler(CircuitTranspilerProtocol):
    """A CircuitTranspiler that converts the gate set of a circuit into the
    specified one.

    Depending on the target gate set and the input circuit, the decomposition may fail
    and an exception may be raised.

    Args:
        target_gateset: A Sequence of allowed output gate names.
    """

    def __init__(
        self,
        target_gateset: Iterable[GateNameType],
        epsilon: float = 1.0e-9,
        validation: bool = True,
    ):
        self._validation = validation
        self._epsilon = epsilon
        self._gateset = set(target_gateset)
        self._target_clifford = cast(
            set[CliffordGateNameType],
            self._gateset & CLIFFORD_GATE_NAMES & SINGLE_QUBIT_GATE_NAMES,
        )
        self._target_rotation = self._gateset & {RX, RY, RZ}
        self._decomposer = self._construct_decomposer()

    def _construct_decomposer(self) -> CircuitTranspiler:
        ts = []

        ts.extend(self._construct_complex_gate_decomposer())
        ts.extend(self._construct_two_qubit_gate_decomposer())

        if self._target_clifford:
            ts.extend(self._construct_rotation_fuser())
            ts.extend(
                [
                    Rotation2NamedTranspiler(epsilon=self._epsilon),  # Optimizer
                    CliffordConversionTranspiler(tuple(self._target_clifford)),
                ]
            )

        if Identity not in self._gateset:
            ts.append(IdentityEliminationTranspiler())

        ts.extend(self._construct_clifford_to_rotation_decomposer())
        ts.extend(self._construct_rotation_fuser())
        ts.append(
            RotationConversionTranspiler(
                target_rotation=tuple(self._target_rotation),
                favorable_clifford=tuple(self._target_clifford),
            )
        )
        ts.extend(self._construct_rotation_fuser())

        return SequentialTranspiler(ts)

    def _construct_rotation_fuser(self) -> list[CircuitTranspiler]:
        return [
            FuseRotationTranspiler(),  # Optimizer
            NormalizeRotationTranspiler(),
            ZeroRotationEliminationTranspiler(epsilon=self._epsilon),  # Optimizer
        ]

    def _construct_complex_gate_decomposer(self) -> list[CircuitTranspiler]:
        return self._collect_decomposers_for_target_gateset(
            {
                Pauli: PauliDecomposeTranspiler(),
                PauliRotation: PauliRotationDecomposeTranspiler(),
                UnitaryMatrix: SequentialTranspiler(
                    [
                        SingleQubitUnitaryMatrix2RYRZTranspiler(),
                        TwoQubitUnitaryMatrixKAKTranspiler(),
                    ]
                ),
                TOFFOLI: TOFFOLI2HTTdagCNOTTranspiler(),
                U1: U1ToRZTranspiler(),
                U2: U2ToRXRZTranspiler(),
                U3: U3ToRXRZTranspiler(),
            }
        )

    def _construct_two_qubit_gate_decomposer(self) -> list[CircuitTranspiler]:
        return self._collect_decomposers_for_target_gateset(
            {
                SWAP: SWAP2CNOTTranspiler(),
                CZ: CZ2CNOTHTranspiler(),
                CNOT: CNOT2CZHTranspiler(),
            }
        )

    def _construct_clifford_to_rotation_decomposer(self) -> list[CircuitTranspiler]:
        return self._collect_decomposers_for_target_gateset(
            {
                H: H2RXRYTranspiler(),
                X: X2RXTranspiler(),
                Y: Y2RYTranspiler(),
                Z: Z2RZTranspiler(),
                SqrtX: SqrtX2RXTranspiler(),
                SqrtXdag: SqrtXdag2RXTranspiler(),
                SqrtY: SqrtY2RYTranspiler(),
                SqrtYdag: SqrtYdag2RYTranspiler(),
                S: S2RZTranspiler(),
                Sdag: Sdag2RZTranspiler(),
                T: T2RZTranspiler(),
                Tdag: Tdag2RZTranspiler(),
            }
        )

    def _collect_decomposers_for_target_gateset(
        self,
        name_decomp_dict: dict[GateNameType, CircuitTranspiler],
    ) -> list[CircuitTranspiler]:
        ts = []
        for name, trans in name_decomp_dict.items():
            if name not in self._gateset:
                ts.append(trans)
        return ts

    def _validate(self, circuit: ImmutableQuantumCircuit) -> None:
        for gate in circuit.gates:
            if gate.name not in self._gateset:
                raise ValueError(f"{gate} cannot be converted into the target gateset.")

    def __call__(self, circuit: ImmutableQuantumCircuit) -> ImmutableQuantumCircuit:
        tr_circuit = self._decomposer(circuit)
        if self._validation:
            self._validate(tr_circuit)
        return tr_circuit
