# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import itertools as it
from collections import defaultdict
from collections.abc import Iterable, MutableMapping, MutableSequence, Sequence
from typing import Callable, Optional

from quri_parts.circuit import QuantumGate

from .noise_instruction import (
    CircuitNoiseInstruction,
    GateNoiseInstruction,
    NoiseInstruction,
    QubitIndices,
    QubitNoisePair,
)


class NoiseModel:
    """Backend independent noise model that can hold multiple
    :class:`NoiseInstruction` conditions.

    First, create :class:`NoiseInstruction` objects to specify the noise type and
    application conditions. Then, given a list of these :class:`NoiseInstruction`,
    :class:`NoiseModel` can be created to represent the noise types and application
    conditions independently of the backend. Finally, when converting to a concrete
    backend, :class:`NoiseModel` is given together with the :class:`QuantumCircuit`
    to generate a circuit to which noise is applied.

    Below is an example of using the Qulacs backend.

    Examples:
        .. highlight:: python
        .. code-block:: python
            from quri_parts.qulacs.circuit.noise import convert_circuit_with_noise_model

            noises = [BitFlipNoise(...), PhaseFlipNoise(...), ...]
            model = NoiseModel(noises)
            circuit = QuantumCircuit(...)
            qulacs_circuit = convert_circuit_with_noise_model(circuit, model)

        :class:`NoiseInstruction` includes :class:`GateNoiseInstruction`, which is
        applied when a specific gate acts, and :class:`CircuitNoiseInstruction`,
        which is applied depending on the circuit structure.

        When creating a :class:`GateNoiseInstruction`, qubit indices and the type of
        gate can be specified in common, along with the parameters of each noise.
        The following explains how to specify the conditions for applying noise,
        using several concrete examples.

        If both qubit_indices and ``target_gates`` are specified, noise is applied
        to any of the gates in ``target_gates``, acting on any of the qubits in
        ``qubit_indices``.

        .. highlight:: python
        .. code-block:: python

            # H or CNOT gate, acting on 1 or 3 index qubit.
            BitFlipNoise(
                error_prob=0.002, qubit_indices=[1, 3], target_gates=["H", "CNOT"]
            )

        An empty list matches everything.

        .. highlight:: python
        .. code-block:: python

            # Any gate, acting on 1 or 3 index qubit.
            BitFlipNoise(error_prob=0.002, qubit_indices=[1, 3], target_gates=[])

            # H or CNOT gate, acting on any qubit.
            BitFlipNoise(error_prob=0.002, qubit_indices=[], target_gates=["H", "CNOT"])

            # Any gate, acting on any qubit.
            BitFlipNoise(error_prob=0.002, qubit_indices=[], target_gates=[])

        For :class:`NoiseInstruction` applied to multiple qubits, qubit_indices must be
        a set equal to qubit indices on which the target gate acts.

        .. highlight:: python
        .. code-block:: python

            # Match with CNOT(0, 2) and CNOT(2, 0).
            PauliNoise([[1, 2], [2, 3]], [0.001, 0.002], [2, 0], ["CNOT"])

            # Match with Pauli(1, 2, 3), Pauli(1, 3, 2), Pauli(2, 1, 3),
            # Pauli(2, 3, 1), Pauli(3, 1, 2), and Pauli(3, 2, 1).
            GeneralDepolarizingNoise(0.004, [1, 2, 3], ["Pauli"])

    Args:
        noises: Sequence of :class:`NoiseInstruction`.
    """

    def __init__(self, noises: Sequence[NoiseInstruction] = []):
        self._id_to_index: MutableMapping[int, int] = {}
        self._index = 0

        self._custom_gate_filters: MutableMapping[
            int, Callable[[QuantumGate], bool]
        ] = {}

        self._gate_noises_for_specified: MutableMapping[
            tuple[str, QubitIndices], MutableSequence[GateNoiseInstruction]
        ] = defaultdict(list)
        self._gate_noises_for_all_gates_specified_qubits: MutableMapping[
            QubitIndices, MutableSequence[GateNoiseInstruction]
        ] = defaultdict(list)
        self._gate_noises_for_all_qubits_specified_gates: MutableMapping[
            str, MutableSequence[GateNoiseInstruction]
        ] = defaultdict(list)
        self._gate_noises_for_all: MutableSequence[GateNoiseInstruction] = []
        self._circuit_noises: MutableSequence[CircuitNoiseInstruction] = []

        self.extend(noises)

    def _issue_index(self) -> int:
        """ID issuer to give serial numbers to NoiseInstructions.

        Returns a serial number that is incremented each time the
        function is called.
        """
        self._index += 1
        return self._index - 1

    def _get_gate_noises_for_all_gates_specified_qubits(
        self, gate: QuantumGate
    ) -> Iterable[QubitNoisePair]:
        """Search for :class:`GateNoiseInstruction` that match the given gate
        from the noises for which qubit indices are specified and gate types
        are not specified."""

        ids = tuple(gate.control_indices) + tuple(gate.target_indices)
        ret: list[tuple[tuple[int, ...], GateNoiseInstruction]] = []

        # NOTE Search for single qubit noises.
        for qubit in ids:
            for noise in self._gate_noises_for_all_gates_specified_qubits.get(
                (qubit,), ()
            ):
                ret.append(((qubit,), noise))

        # NOTE Search for multiple qubit noises.
        if len(ids) > 1:
            for noise in self._gate_noises_for_all_gates_specified_qubits.get(
                tuple(sorted(ids)),
                (),
            ):
                ret.append((ids, noise))

        return ret

    def _get_gate_noises_for_all_qubits_specified_gates(
        self, gate: QuantumGate
    ) -> Iterable[QubitNoisePair]:
        """Search for :class:`GateNoiseInstruction` that match the given gate
        from the noises for which qubit indices are not specified and gate
        types are specified."""
        return self._get_gate_noises_for_all(
            gate, self._gate_noises_for_all_qubits_specified_gates.get(gate.name, ())
        )

    def _get_gate_noises_for_all(
        self, gate: QuantumGate, noises: Sequence[GateNoiseInstruction]
    ) -> Iterable[QubitNoisePair]:
        """Search for :class:`GateNoiseInstruction` that match the given gate
        from the noises for which neither qubit indices nor gate types are
        specified."""

        ids = tuple(gate.control_indices) + tuple(gate.target_indices)
        ret: list[tuple[tuple[int, ...], GateNoiseInstruction]] = []

        for noise in noises:
            if noise.qubit_count == 1:  # NOTE Single qubit noise.
                ret.extend([((qubit,), noise) for qubit in ids])
            elif noise.qubit_count == len(ids):  # NOTE Multiple qubit noise.
                ret.append((ids, noise))
        return ret

    def _get_gate_noises_for_specified(
        self, gate: QuantumGate
    ) -> Iterable[QubitNoisePair]:
        """Search for :class:`GateNoiseInstruction` that match the given gate
        from the noises for which both qubit indices and gate types are
        specified."""

        ids = tuple(gate.control_indices) + tuple(gate.target_indices)
        ret: list[tuple[tuple[int, ...], GateNoiseInstruction]] = []

        # NOTE Search for single qubit noises.
        for qubit in ids:
            for noise in self._gate_noises_for_specified.get((gate.name, (qubit,)), ()):
                ret.append(((qubit,), noise))

        # NOTE Search for multiple qubit noises.
        if len(ids) > 1:
            for noise in self._gate_noises_for_specified.get(
                (gate.name, tuple(sorted(ids))),
                (),
            ):
                ret.append((ids, noise))
        return ret

    def noises_for_gate(self, gate: QuantumGate) -> Iterable[QubitNoisePair]:
        """For a given Gate, search for matching qubit indices and
        :class:`NoiseInstruction` pairs.

        Returns a list of pairs of qubit indices and :class:`NoiseInstruction`
        that match the given gate conditions in this noise model. The order of
        the list is the order in which :class:`NoiseInstructions` are added to
        the noise model.

        Args:
           gate: :class:`QuantumGate` to which noises are applied.
        """
        cpairs = it.chain(
            self._get_gate_noises_for_all(gate, self._gate_noises_for_all),
            self._get_gate_noises_for_all_qubits_specified_gates(gate),
            self._get_gate_noises_for_all_gates_specified_qubits(gate),
            self._get_gate_noises_for_specified(gate),
        )

        if self._custom_gate_filters:
            pairs = tuple(
                filter(
                    lambda pair: id(pair[1]) not in self._custom_gate_filters
                    or self._custom_gate_filters[id(pair[1])](gate),
                    cpairs,
                )
            )
        else:
            pairs = tuple(cpairs)

        return sorted(pairs, key=lambda x: self._id_to_index[id(x[1])])

    def noises_for_circuit(self) -> Sequence[CircuitNoiseInstruction]:
        """Returns sequence of :class:`CircuitNoiseInstruction` in the
        model."""
        return self._circuit_noises

    def _add_gate_noise_for_all_gates_specified_qubits(
        self, noise: GateNoiseInstruction
    ) -> None:
        """Add :class:`GateNoiseInstruction`, such that qubit indices are
        specified and gate types are not specified, to the internal data
        structure in easily searchable form."""

        # NOTE Multiple qubit noise is selected by an exact match of a set of qubits,
        # while single qubit noise is selected by a match with any of the registered
        # qubits.
        if noise.qubit_count > 1:
            qcs = [tuple(sorted(noise.qubit_indices))]
        else:
            qcs = [(q,) for q in noise.qubit_indices]

        for qubits in qcs:
            self._gate_noises_for_all_gates_specified_qubits[qubits].append(noise)

    def _add_gate_noise_for_all_qubits_specified_gates(
        self, noise: GateNoiseInstruction
    ) -> None:
        """Add :class:`GateNoiseInstruction`, such that qubit indices are not
        specified and gate types are specified, to the internal data structure
        in easily searchable form."""
        for gate in noise.target_gates:
            self._gate_noises_for_all_qubits_specified_gates[gate].append(noise)

    def _add_gate_noise_for_all(self, noise: GateNoiseInstruction) -> None:
        """Add :class:`GateNoiseInstruction`, which matches all, with neither
        qubit indices nor gate types are specified, to the internal data
        structure."""
        self._gate_noises_for_all.append(noise)

    def _add_gate_noise_for_specified(self, noise: GateNoiseInstruction) -> None:
        """Add :class:`GateNoiseInstruction`, in which both qubit indices and
        gate types are specified, to the internal data structure in easily
        searchable form."""

        # NOTE Multiple qubit noise is selected by an exact match of a set of qubits,
        # while single qubit noise is selected by a match with any of the registered
        # qubits.
        if noise.qubit_count > 1:
            qcs = [tuple(sorted(noise.qubit_indices))]
        else:
            qcs = [(q,) for q in noise.qubit_indices]

        for qubits in qcs:
            for gate in noise.target_gates:
                self._gate_noises_for_specified[(gate, qubits)].append(noise)

    def _add_circuit_noise(self, noise: CircuitNoiseInstruction) -> None:
        """Add :class:`CircuitNoiseInstruction` to the internal data
        structure."""
        self._circuit_noises.append(noise)

    def add_noise(
        self,
        noise: NoiseInstruction,
        custom_gate_filter: Optional[Callable[[QuantumGate], bool]] = None,
    ) -> None:
        """Add single :class:`NoiseInstruction` to the model and update the
        state of the model.

        Args:
            noise: :class:`NoiseInstruction` to be added to the model.
            custom_gate_filter: (If specified) Additional condition to determine
                if the noise is applied to the target gate.
        """

        self._id_to_index[id(noise)] = self._issue_index()

        if custom_gate_filter is not None:
            self._custom_gate_filters[id(noise)] = custom_gate_filter

        if isinstance(noise, GateNoiseInstruction):
            if (not noise.qubit_indices) and (not noise.target_gates):
                self._add_gate_noise_for_all(noise)
            elif not noise.qubit_indices:
                self._add_gate_noise_for_all_qubits_specified_gates(noise)
            elif not noise.target_gates:
                self._add_gate_noise_for_all_gates_specified_qubits(noise)
            else:
                self._add_gate_noise_for_specified(noise)
        elif isinstance(noise, CircuitNoiseInstruction):
            self._add_circuit_noise(noise)
        else:
            raise ValueError(f"Unsupported NoiseInstuction type: {type(noise)}")

    def extend(self, noises: Sequence[NoiseInstruction]) -> None:
        """Add multiple :class:`NoiseInstruction` to the model and update the
        state of the model."""
        for noise in noises:
            self.add_noise(noise)

    def __iadd__(self, noises: Sequence[NoiseInstruction]) -> "NoiseModel":
        """Addition assignment operator to add multiple
        :class:`NoiseInstruction` to the model."""
        self.extend(noises)
        return self

    def __eq__(self, other: object) -> bool:
        """Object equivalence testing with comparison of all internal
        states."""
        if not isinstance(other, NoiseModel):
            return False
        return (
            self._gate_noises_for_specified,
            self._gate_noises_for_all_gates_specified_qubits,
            self._gate_noises_for_all_qubits_specified_gates,
            self._gate_noises_for_all,
            self._circuit_noises,
            self._index,
            self._id_to_index,
            self._custom_gate_filters,
        ) == (
            other._gate_noises_for_specified,
            other._gate_noises_for_all_gates_specified_qubits,
            other._gate_noises_for_all_qubits_specified_gates,
            other._gate_noises_for_all,
            other._circuit_noises,
            other._index,
            other._id_to_index,
            other._custom_gate_filters,
        )
