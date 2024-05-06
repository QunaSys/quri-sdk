# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import Counter
from typing import TYPE_CHECKING, Any, Iterable, Optional, Union

import numpy as np
import qulacs as ql
from numpy import cfloat, zeros
from numpy.typing import NDArray

from quri_parts.circuit import NonParametricQuantumCircuit
from quri_parts.core.sampling import (
    ConcurrentStateSampler,
    MeasurementCounts,
    StateSampler,
    ideal_sample_from_state_vector,
    sample_from_state_vector,
)
from quri_parts.core.state import CircuitQuantumState, QuantumStateVector
from quri_parts.core.utils.concurrent import execute_concurrently
from quri_parts.qulacs.circuit import convert_circuit
from quri_parts.qulacs.circuit.compiled_circuit import _QulacsCircuit

from . import QulacsStateT, cast_to_list

if TYPE_CHECKING:
    from concurrent.futures import Executor


def _evaluate_qp_state_to_qulacs_state(state: QulacsStateT) -> ql.QuantumState:
    n_qubits = state.qubit_count

    if isinstance(state, QuantumStateVector):
        init_state_vector = state.vector
    elif isinstance(state, CircuitQuantumState):
        init_state_vector = zeros(2**n_qubits, dtype=complex)
        init_state_vector[0] = 1.0
    else:
        raise TypeError(
            "the input state should be either a GeneralCircuitQuantumState\
             or a QuantumStateVector"
        )
    return _get_updated_qulacs_state_from_vector(state.circuit, init_state_vector)


def _get_updated_qulacs_state_from_vector(
    circuit: Union[NonParametricQuantumCircuit, _QulacsCircuit],
    init_state: NDArray[cfloat],
) -> ql.QuantumState:
    if len(init_state) != 2**circuit.qubit_count:
        raise ValueError("Inconsistent qubit length between circuit and state")

    qulacs_state = ql.QuantumState(circuit.qubit_count)
    qulacs_state.load(cast_to_list(init_state))

    if isinstance(circuit, _QulacsCircuit):
        qulacs_cicuit = circuit._qulacs_circuit
    else:
        qulacs_cicuit = convert_circuit(circuit)

    qulacs_cicuit.update_quantum_state(qulacs_state)

    return qulacs_state


def evaluate_state_to_vector(state: QulacsStateT) -> QuantumStateVector:
    """Convert GeneralCircuitQuantumState or QuantumStateVector to
    QuantumStateVector that only contains the state vector."""
    out_state_vector = _evaluate_qp_state_to_qulacs_state(state)

    # We need to disable type check due to an error in qulacs type annotation
    # https://github.com/qulacs/qulacs/issues/537
    return QuantumStateVector(
        state.qubit_count, out_state_vector.get_vector()  # type: ignore
    )


def run_circuit(
    circuit: NonParametricQuantumCircuit,
    init_state: NDArray[cfloat],
) -> NDArray[cfloat]:
    """Act a NonParametricQuantumCircuit onto a state vector and returns a new
    state vector."""

    qulacs_state = _get_updated_qulacs_state_from_vector(circuit, init_state)
    # We need to disable type check due to an error in qulacs type annotation
    # https://github.com/qulacs/qulacs/issues/537
    new_state_vector: NDArray[cfloat] = qulacs_state.get_vector()  # type: ignore

    return new_state_vector


def get_marginal_probability(
    state_vector: NDArray[cfloat], measured_values: dict[int, int]
) -> float:
    """Compute the probability of obtaining a result when measuring on a subset
    of the qubits.

    state_vector:
        A 1-dimensional array representing the state vector.
    measured_values:
        A dictionary representing the desired measurement outcome on the specified
        qubtis. Suppose {0: 1, 2: 0} is passed in, it computes the probability of
        obtaining 1 on the 0th qubit and 0 on the 2nd qubit.
    """
    n_qubits: float = np.log2(state_vector.shape[0])
    assert n_qubits.is_integer(), "Length of the state vector must be a power of 2."
    assert (
        max(measured_values.keys()) < n_qubits
    ), f"The specified qubit index {max(measured_values.keys())} is out of range."

    qulacs_state = ql.QuantumState(int(n_qubits))
    qulacs_state.load(cast_to_list(state_vector))
    measured = [measured_values.get(i, 2) for i in range(int(n_qubits))]
    return qulacs_state.get_marginal_probability(measured)


def create_qulacs_vector_state_sampler() -> StateSampler[QulacsStateT]:
    """Creates a state sampler based on Qulacs circuit execution."""

    def state_sampler(state: QulacsStateT, n_shots: int) -> MeasurementCounts:
        if n_shots > 2 ** max(state.qubit_count, 10):
            # Use multinomial distribution for faster sampling
            state_vector = evaluate_state_to_vector(state).vector
            return sample_from_state_vector(state_vector, n_shots)

        qs_state = _evaluate_qp_state_to_qulacs_state(state)
        return Counter(qs_state.sampling(n_shots))

    return state_sampler


def _sequential_vector_state_sampler(
    _: Any, state_shots_tuples: Iterable[tuple[QulacsStateT, int]]
) -> Iterable[MeasurementCounts]:
    state_sampler = create_qulacs_vector_state_sampler()
    return [state_sampler(state, shots) for state, shots in state_shots_tuples]


def create_concurrent_vector_state_sampler(
    executor: Optional["Executor"] = None, concurrency: int = 1
) -> ConcurrentStateSampler[QulacsStateT]:
    def concurrent_state_sampler(
        state_shots_tuples: Iterable[tuple[QulacsStateT, int]]
    ) -> Iterable[MeasurementCounts]:
        return execute_concurrently(
            _sequential_vector_state_sampler,
            None,
            state_shots_tuples,
            executor,
            concurrency,
        )

    return concurrent_state_sampler


def create_qulacs_ideal_vector_state_sampler() -> StateSampler[QulacsStateT]:
    """Creates an ideal state sampler based on Qulacs circuit execution."""

    def ideal_state_sampler(
        state: Union[CircuitQuantumState, QuantumStateVector], n_shots: int
    ) -> MeasurementCounts:
        state_vector = evaluate_state_to_vector(state).vector
        return ideal_sample_from_state_vector(state_vector, n_shots)

    return ideal_state_sampler
