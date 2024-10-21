# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections.abc import Iterable
from typing import TYPE_CHECKING, cast

import numpy as np
from scipy import linalg

from quri_parts.circuit import ImmutableQuantumCircuit
from quri_parts.core.sampling import (
    ConcurrentSampler,
    MeasurementCounts,
    Sampler,
    create_sampler_from_concurrent_sampler,
)
from quri_parts.core.state import ComputationalBasisState

if TYPE_CHECKING:
    import numpy.typing as npt  # noqa: F401


def create_filter_matrix(
    qubit_count: int, sampler: ConcurrentSampler, shots: int
) -> "npt.NDArray[np.float64]":
    """Generate a filter matrix for readout mitigation.

    Args:
        qubit_count: Number of qubits of the target circuit.
        sampler: :class:`ConcurrentSampler` for the target backend.
        shots: Number of shots at each sampling.
    """
    dim = 2**qubit_count
    pairs = [
        (ComputationalBasisState(n_qubits=qubit_count, bits=n).circuit, shots)
        for n in range(dim)
    ]
    amatrix = []
    for counts in sampler(pairs):
        cv: "npt.NDArray[np.float64]" = np.array(
            [float(counts[k]) if k in counts else 0.0 for k in range(dim)]
        )
        amatrix.append(cv / linalg.norm(cv, ord=1))
    return cast("npt.NDArray[np.float64]", linalg.pinv(np.array(amatrix).T))


def readout_mitigation(
    counts: Iterable[MeasurementCounts],
    filter_matrix: "npt.NDArray[np.float64]",
) -> Iterable[MeasurementCounts]:
    """Apply readout mitigation to the sampling result.

    Args:
        counts: Sampling result as a :class:`MeasurementCounts`.
        filter_matrix: Filter matrix for the target circuit created by the
            ``create_filter_matrix`` function.
    """
    if filter_matrix.ndim != 2 or filter_matrix.shape[0] != filter_matrix.shape[1]:
        raise ValueError(
            f"The shape of the filter matrix is incorrect: {filter_matrix.shape}."
        )

    def counts_iter() -> Iterable[MeasurementCounts]:
        for count in counts:
            dim = filter_matrix.shape[0]
            cnoisy: "npt.NDArray[np.float64]" = np.array(
                [float(count[k]) if k in count else 0.0 for k in range(dim)]
            )
            cideal = filter_matrix.dot(cnoisy)
            yield {k: cideal[k] for k in range(dim) if cideal[k] > 0.0}

    return counts_iter()


def create_readout_mitigation_concurrent_sampler(
    qubit_count: int, sampler: ConcurrentSampler, shots: int
) -> ConcurrentSampler:
    """Wrap the given :class:`ConcurrentSampler` to create a
    :class:`ConcurrentSampler` where readout mitigation is automatically
    applied to the result.

    Args:
        qubit_count: Number of qubits of the target circuit.
        sampler: :class:`ConcurrentSampler` for the target backend.
        shots: Number of shots for each sampling to create the filter matrix.
    """
    filter_matrix = create_filter_matrix(qubit_count, sampler, shots)

    def wrapped_sampler(
        pairs: Iterable[tuple[ImmutableQuantumCircuit, int]]
    ) -> Iterable[MeasurementCounts]:
        return readout_mitigation(sampler(pairs), filter_matrix)

    return wrapped_sampler


def create_readout_mitigation_sampler(
    qubit_count: int, sampler: ConcurrentSampler, shots: int
) -> Sampler:
    """Wrap the given :class:`ConcurrentSampler` to create a
    :class:`Sampler` where readout mitigation is automatically
    applied to the result.

    Args:
        qubit_count: Number of qubits of the target circuit.
        sampler: :class:`ConcurrentSampler` for the target backend.
        shots: Number of shots for each sampling to create the filter matrix.
    """

    mitigate_sampler = create_readout_mitigation_concurrent_sampler(
        qubit_count, sampler, shots
    )
    return create_sampler_from_concurrent_sampler(mitigate_sampler)
