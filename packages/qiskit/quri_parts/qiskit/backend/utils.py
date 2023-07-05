# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Callable, Mapping, MutableMapping, Optional, Sequence

from qiskit.providers.backend import Backend, BackendV1, BackendV2

from quri_parts.backend import BackendError, SamplingCounts, SamplingJob
from quri_parts.backend.qubit_mapping import BackendQubitMapping, QubitMappedSamplingJob
from quri_parts.circuit.transpile import CircuitTranspiler, SequentialTranspiler
from quri_parts.qiskit.circuit import QiskitTranspiler


def distribute_backend_shots(
    n_shots: int,
    min_shots: int,
    max_shots: Optional[int],
    enable_shots_roundup: Optional[bool] = True,
) -> Sequence[int]:
    """Distributes the n_shots into batches of smaller shot numbers according
    to the max_shots value.

    Args:
        n_shots: Total number of shots.
        min_shots: Minimal number of shots of a single batch.
        max_shots: Maximal number of shots of a single batch.
        enable_shots_roundup: If True, when a number of shots of a batch
            is smaller than min_shots, it is rounded up to the minimum.
            In this case, it is possible that shots more than specified are used.
            If it is strictly not allowed to exceed the specified shot count,
            set this argument to False.
    """
    if max_shots is not None and n_shots > max_shots:
        shot_dist = [max_shots] * (n_shots // max_shots)
        remaining = n_shots % max_shots
        if remaining > 0:
            if remaining >= min_shots:
                shot_dist.append(remaining)
            elif enable_shots_roundup:
                shot_dist.append(min_shots)
    else:
        if n_shots >= min_shots or enable_shots_roundup:
            shot_dist = [max(n_shots, min_shots)]
        else:
            raise ValueError(
                f"n_shots is smaller than minimum shot count ({min_shots}) "
                "supported by the device. Try larger n_shots or use "
                "enable_shots_roundup=True when creating the backend."
            )
    return shot_dist


def get_backend_min_max_shot(backend: Backend) -> tuple[int, Optional[int]]:
    """Get the selected qiskit backend's minimum and maximum shot number
    allowed in a single sampling job."""
    if not isinstance(backend, (BackendV1, BackendV2)):
        raise BackendError("Backend not supported.")

    if isinstance(backend, BackendV1):
        max_shots = backend.configuration().max_shots
        if max_shots > 0:
            return 1, max_shots

    return 1, backend.max_shots


def get_job_mapper_and_circuit_transpiler(
    qubit_mapping: Optional[Mapping[int, int]] = None,
    circuit_transpiler: Optional[CircuitTranspiler] = None,
) -> tuple[Callable[[SamplingJob], SamplingJob], CircuitTranspiler]:
    """Creates
        1. A job mapper that maps the qubits of raw sampling job from the backend to
            a :class:`~SamplingJob` according to the specified qubit_mapping.
        2. A circuit transpiler.

    Args:
        qubit_mapping: If specified, indices of qubits in the circuit are remapped
            before running it on the backend. It can be used when you want to use
            specific backend qubits, e.g. those with high fidelity.
            The mapping should be specified with "from" qubit
            indices as keys and "to" qubit indices as values. For example, if
            you want to map qubits 0, 1, 2, 3 to backend qubits as 0 → 4, 1 → 2,
            2 → 5, 3 → 0, then the ``qubit_mapping`` should be
            ``{0: 4, 1: 2, 2: 5, 3: 0}``.
        circuit_transpiler: A transpiler applied to the circuit before running it.
            :class:`~QiskitTranspiler` is used when not specified.
    """
    if circuit_transpiler is None:
        circuit_transpiler = QiskitTranspiler()

    if qubit_mapping:
        mapper = BackendQubitMapping(qubit_mapping)
        circuit_transpiler = SequentialTranspiler(
            [circuit_transpiler, mapper.circuit_transpiler]
        )
        composite_job_qubit_mapper: Callable[
            [SamplingJob], SamplingJob
        ] = lambda job: QubitMappedSamplingJob(
            job, mapper
        )  # noqa: E731
        return composite_job_qubit_mapper, circuit_transpiler
    else:
        simple_job_qubit_mapper: Callable[
            [SamplingJob], SamplingJob
        ] = lambda job: job  # noqa: E731
        return simple_job_qubit_mapper, circuit_transpiler


def convert_qiskit_sampling_count_to_qp_sampling_count(
    qiskit_counts: Mapping[str, int]
) -> SamplingCounts:
    """Converts the raw counter returned from qiskit backends to quri-parts
    couter.

    Note that the qiskit counter uses a string that represents a binary
    number as the key.
    """
    measurements: MutableMapping[int, int] = {}
    for result in qiskit_counts:
        measurements[int(result, 2)] = qiskit_counts[result]
    return measurements
