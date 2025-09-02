from collections.abc import Mapping, Sequence
from typing import Callable, Optional, Union

from .circuit import ImmutableQuantumCircuit
from .gate import QuantumGate

class GateNoiseInstruction:
    def __new__(
        cls,
        name: str,
        qubit_count: int,
        params: Sequence[float],
        qubit_indices: Sequence[int],
        target_gates: Sequence[str],
        pauli_list: Sequence[Sequence[int]] = (),
        prob_list: Sequence[float] = (),
        kraus_operators: Sequence[Sequence[Sequence[float]]] = (),
        gate_matrices: Sequence[Sequence[Sequence[float]]] = (),
    ) -> "GateNoiseInstruction": ...
    @property
    def name(self) -> str: ...
    @property
    def qubit_count(self) -> int: ...
    @property
    def params(self) -> Sequence[float]: ...
    @property
    def qubit_indices(self) -> Sequence[int]: ...
    @property
    def target_gates(self) -> Sequence[str]: ...
    @property
    def pauli_list(self) -> Sequence[Sequence[int]]: ...
    @property
    def prob_list(self) -> Sequence[float]: ...
    @property
    def kraus_operators(self) -> Sequence[Sequence[Sequence[float]]]: ...
    @property
    def gate_matrices(self) -> Sequence[Sequence[Sequence[float]]]: ...
    def __eq__(self, other: object) -> bool: ...

class GateIntervalNoise:
    def __new__(
        cls, noises: Sequence[GateNoiseInstruction], gate_interval: int
    ) -> "GateIntervalNoise": ...
    def name(self) -> str: ...

class DepthIntervalNoise:
    def __new__(
        cls, noises: Sequence[GateNoiseInstruction], depth_interval: int
    ) -> "DepthIntervalNoise": ...
    def name(self) -> str: ...

class MeasurementNoise:
    def __new__(
        cls, noises: Sequence[GateNoiseInstruction], qubit_indices: Sequence[int] = ()
    ) -> "MeasurementNoise": ...
    def name(self) -> str: ...

class CircuitNoiseInstance:
    def noises_for_gate(
        self, gate: QuantumGate, circuit: ImmutableQuantumCircuit
    ) -> Sequence[tuple[Sequence[int], GateNoiseInstruction]]: ...
    def noises_for_depth(
        self, qubits: Sequence[int], circuit: ImmutableQuantumCircuit
    ) -> Sequence[tuple[Sequence[int], GateNoiseInstruction]]: ...

class NoiseModel:
    def __new__(
        cls,
        noises: Sequence[
            Union[
                GateNoiseInstruction,
                GateIntervalNoise,
                DepthIntervalNoise,
                MeasurementNoise,
            ]
        ] = [],
    ) -> "NoiseModel": ...
    def gate_noise_from_id(self, noise_id: int) -> GateNoiseInstruction: ...
    def gate_noise_list(self) -> Sequence[GateNoiseInstruction]: ...
    def noises_for_gate(
        self, gate: QuantumGate
    ) -> Sequence[tuple[Sequence[int], GateNoiseInstruction]]: ...
    def noises_for_circuit(self) -> CircuitNoiseInstance: ...
    def add_gate_interval_noise(self, noise: GateIntervalNoise) -> None: ...
    def add_depth_interval_noise(self, noise: DepthIntervalNoise) -> None: ...
    def add_measurement_noise(self, noise: MeasurementNoise) -> None: ...
    def add_gate_noise(self, noise: GateNoiseInstruction) -> int: ...
    def add_noise(
        self,
        noise: Union[
            GateNoiseInstruction,
            GateIntervalNoise,
            DepthIntervalNoise,
            MeasurementNoise,
        ],
        custom_gate_filter: Optional[Callable[[QuantumGate], bool]] = None,
    ) -> None: ...
    def extend(
        self,
        noises: Sequence[
            Union[
                GateNoiseInstruction,
                GateIntervalNoise,
                DepthIntervalNoise,
                MeasurementNoise,
            ]
        ],
    ) -> None: ...
