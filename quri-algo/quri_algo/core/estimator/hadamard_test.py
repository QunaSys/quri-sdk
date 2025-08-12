# Licensed under the MIT License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      https://mit-license.org/
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import defaultdict
from functools import reduce
from typing import Any, NamedTuple, Union, cast

import numpy as np
from quri_parts.circuit import NonParametricQuantumCircuit, QuantumCircuit
from quri_parts.circuit.transpile import CircuitTranspiler, QubitRemappingTranspiler
from quri_parts.core.estimator import Estimate
from quri_parts.core.sampling import MeasurementCounts, Sampler, StateSampler
from quri_parts.core.state import (
    GeneralCircuitQuantumState,
    QuantumStateT,
    QuantumStateVector,
)

from quri_algo.circuit.hadamard_test import HadamardTestCircuitFactory
from quri_algo.circuit.interface import CircuitFactory

from .interface import ExpectationValueEstimator, State, StateT


class _Estimate(NamedTuple):
    value: complex
    error: float = np.nan


class HadamardTestMeasurementCount(NamedTuple):
    real_sampling_cnt: MeasurementCounts
    imag_sampling_cnt: MeasurementCounts

    @property
    def expectation_value(self) -> Estimate[complex]:
        real = get_expectation_val_from_hadamard_counter(self.real_sampling_cnt)
        imag = get_expectation_val_from_hadamard_counter(self.imag_sampling_cnt)
        return _Estimate(value=real + 1j * imag)


def _general_sample_on_state(
    sampler: Union[Sampler, StateSampler[State]], state: State, n_shots: int
) -> MeasurementCounts:
    if isinstance(state, QuantumStateVector):
        sampler = cast(StateSampler[State], sampler)
        return sampler(state, n_shots)
    sampler = cast(Sampler, sampler)
    return sampler(state.circuit, n_shots)


class HadamardTest(ExpectationValueEstimator[StateT]):
    r"""Estimate a unitary operator U's expectation value :math:`\langle U.

    \rangle` with Hadamard test.
    """

    def __init__(
        self,
        controlled_circuit_factory: CircuitFactory,
        sampler: Union[Sampler, StateSampler[StateT]],
        *,
        transpiler: CircuitTranspiler | None = None
    ):
        self.controlled_circuit_factory = controlled_circuit_factory
        self.sampler = sampler
        self.transpiler = transpiler

    @property
    def real_circuit_factory(self) -> HadamardTestCircuitFactory:
        return HadamardTestCircuitFactory(
            True,
            self.controlled_circuit_factory,
            transpiler=self.transpiler,
        )

    @property
    def imag_circuit_factory(self) -> HadamardTestCircuitFactory:
        return HadamardTestCircuitFactory(
            False,
            self.controlled_circuit_factory,
            transpiler=self.transpiler,
        )

    def __call__(
        self, state: StateT, n_shots: int, *args: Any, **kwd: Any
    ) -> Estimate[complex]:
        # TODO: Break into smaller pieces
        input_state = remap_state_for_hadamard_test(state)  # type: ignore
        real_hadamard_state = input_state.with_gates_applied(
            self.real_circuit_factory(*args, **kwd)
        )
        imag_hadamard_state = input_state.with_gates_applied(
            self.imag_circuit_factory(*args, **kwd)
        )

        # TODO: Fix after GeneralSampler is available in QURI Parts
        real_cnt = _general_sample_on_state(self.sampler, real_hadamard_state, n_shots)  # type: ignore
        imag_cnt = _general_sample_on_state(self.sampler, imag_hadamard_state, n_shots)  # type: ignore

        real_cnt = get_hadamard_test_ancilla_qubit_counter(real_cnt)
        imag_cnt = get_hadamard_test_ancilla_qubit_counter(imag_cnt)
        return HadamardTestMeasurementCount(real_cnt, imag_cnt).expectation_value


def shift_state_circuit(
    state_circuit: NonParametricQuantumCircuit, shift: int = 1
) -> NonParametricQuantumCircuit:
    n_qubits = state_circuit.qubit_count
    shifted_circuit = QuantumCircuit(n_qubits + shift, gates=state_circuit.gates)
    remap = {i: i + shift for i in range(n_qubits)}
    remapping_transpiler = QubitRemappingTranspiler(remap)
    return remapping_transpiler(shifted_circuit)


def remap_state_for_hadamard_test(
    state: QuantumStateT, shift: int = 1
) -> QuantumStateT:
    """Shift the input state by 1 qubit for to conform with the Hadamard test
    convention."""
    n_hadamard_test_qubit = state.qubit_count + shift
    circuit = shift_state_circuit(state.circuit, shift)
    if isinstance(state, QuantumStateVector):
        padding = np.zeros(2**shift, dtype=np.complex128)
        padding[0] = 1.0
        state_list = [state.vector]
        state_list.extend([padding])
        vector = reduce(np.kron, state_list)
        return QuantumStateVector(n_hadamard_test_qubit, vector=vector, circuit=circuit)
    return GeneralCircuitQuantumState(n_hadamard_test_qubit, circuit=circuit)


def get_hadamard_test_ancilla_qubit_counter(
    cnt: MeasurementCounts,
) -> MeasurementCounts:
    """Convert all measurement count to ancilla qubit measurement count."""
    hadamard_cnt: defaultdict[int, Union[int, float]] = defaultdict(int)
    for c in cnt:
        hadamard_cnt[c % 2] += cnt[c]
    return cast(MeasurementCounts, hadamard_cnt)


def get_expectation_val_from_hadamard_counter(hadamard_cnt: MeasurementCounts) -> float:
    return (hadamard_cnt[0] - hadamard_cnt[1]) / sum(hadamard_cnt.values())


def parse_hadamard_test_result(cnt: MeasurementCounts) -> float:
    hadamard_cnt = get_hadamard_test_ancilla_qubit_counter(cnt)
    return get_expectation_val_from_hadamard_counter(hadamard_cnt)
