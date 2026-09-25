from collections import Counter
from typing import Callable, List, Mapping, Optional, Sequence, Union

import numpy as np
from numpy.typing import NDArray

from .circuit import GateSequence, ImmutableQuantumCircuit, QuantumCircuit
from .gate import QuantumGate

_MeasurementCounts = Mapping[int, Union[int, float]]
_Sampler = Callable[[ImmutableQuantumCircuit, int], _MeasurementCounts]


class ClassicalState:
    """A :class: for sampling with classical simulation."""
    def __init__(
        self, n_qubits: int, circuit: Optional[ImmutableQuantumCircuit] = None
    ) -> None:
        self._n_qubits: int = n_qubits
        if circuit is None:
            self._circuit = QuantumCircuit(n_qubits).freeze()
        else:
            if circuit.qubit_count != n_qubits:
                raise ValueError(
                    f"n_qubits={n_qubits} does not match with circuit.qubit_count="
                    f"{circuit.qubit_count}"
                )
            self._circuit = circuit.freeze()

    @property
    def qubit_count(self) -> int:
        return self._n_qubits

    @property
    def circuit(self) -> ImmutableQuantumCircuit:
        return self._circuit

    def with_gates_applied(self, gates: GateSequence) -> "ClassicalState":
        circuit = self.circuit + gates
        return ClassicalState(self._n_qubits, circuit)

    def sample(self, n_shots: int) -> _MeasurementCounts:
        gates: Sequence[QuantumGate] = self.circuit.gates
        state: NDArray[np.bool_] = np.zeros(self._n_qubits, dtype=np.bool_)
        for gate in gates:
            _apply_gate_to_classical_array(gate, state)
        state_idx: int = 0
        for i, bit in enumerate(state):
            if bit:
                state_idx += 1 << i
        return Counter({state_idx: n_shots})


def _apply_gate_to_classical_array(gate: QuantumGate, state: NDArray[np.bool_]) -> None:
    if gate.name == "X":
        tgt = gate.target_indices[0]
        state[tgt] ^= True
    elif gate.name == "CNOT":
        tgt = gate.target_indices[0]
        ctrl = gate.control_indices[0]
        if state[ctrl]:
            state[tgt] ^= True
    elif gate.name == "SWAP":
        tgt1, tgt2 = gate.target_indices[0:2]
        state[tgt1], state[tgt2] = state[tgt2], state[tgt1]
    elif gate.name == "TOFFOLI":
        tgt = gate.target_indices[0]
        ctrl1, ctrl2 = gate.control_indices[0:2]
        if state[ctrl1] and state[ctrl2]:
            state[tgt] ^= True
    elif gate.name == "MCX":
        tgt = gate.target_indices[0]
        ctrls: List[bool] = [state[idx] for idx in gate.control_indices]
        if all(ctrls):
            state[tgt] ^= True
    elif gate.name == "Pauli":
        for tgt, pid in zip(gate.target_indices, gate.pauli_ids):
            if pid == 0 or pid == 3:
                # I or Z: diagonal in computational basis, no bit flip.
                continue
            if pid == 1:
                state[tgt] ^= True
            else:
                raise ValueError(
                    f"Pauli gate with pauli_id={pid} is not supported in "
                    "ClassicalSimulator (only Identity, X, and Z are allowed)."
                )
    elif gate.name in [
        "Z",
        "RZ",
        "CZ",
        "U1",
        "S",
        "Sdag",
        "T",
        "Tdag",
        "Identity",
        "Measurement",
        "MCZ",
        "MCS",
        "MCSdag",
        "MCT",
        "MCTdag",
        "MCRZ",
        "MCU1",
    ]:
        pass
    else:
        raise ValueError(f"{gate.name} is not supported in ClassicalSimulator.")


def _state_sampler(state: ClassicalState, n_shots: int) -> _MeasurementCounts:
    return state.sample(n_shots)


def _sampler(circuit: ImmutableQuantumCircuit, n_shots: int) -> _MeasurementCounts:
    return _state_sampler(ClassicalState(circuit.qubit_count, circuit), n_shots)


def create_classical_state_sampler() -> (
    Callable[[ClassicalState, int], _MeasurementCounts]
):
    """Returns a function that performs on `ClassicalState`."""
    return _state_sampler


def create_classical_sampler() -> _Sampler:
    """Returns a function that uses classical simulator for sampling.
    The function conforms to the interface of `quri_parts.core.sampling.Sampler`,
    but is not an instance of it due to circular dependencies."""
    return _sampler
