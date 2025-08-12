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
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, Optional

import juliacall
from juliacall import Main as jl

from quri_parts.circuit import ImmutableQuantumCircuit
from quri_parts.core.sampling import ConcurrentSampler, MeasurementCounts, Sampler
from quri_parts.core.utils.concurrent import execute_concurrently
from quri_parts.itensor.circuit import convert_circuit
from quri_parts.itensor.load_itensor import ensure_itensor_loaded

if TYPE_CHECKING:
    from concurrent.futures import Executor


def _sample(
    circuit: ImmutableQuantumCircuit, shots: int, **kwargs: Any
) -> MeasurementCounts:
    qubits = circuit.qubit_count
    s: juliacall.VectorValue = jl.siteinds("Qubit", qubits)
    psi: juliacall.AnyValue = jl.init_state(s, qubits)
    if len(circuit.gates) == 0:
        return Counter({0: shots})
    circuit_ops = convert_circuit(circuit, s)
    psi = jl.apply(circuit_ops, psi, **kwargs)
    if any(k in kwargs for k in ["mindim", "maxdim", "cutoff"]):
        psi = jl.normalize(psi)

    result: list[int] = jl.sampling(psi, shots)
    return Counter(result)


def create_itensor_mps_sampler(
    *,
    maxdim: Optional[int] = None,
    cutoff: Optional[float] = None,
    **kwargs: Any,
) -> Sampler:
    """Returns a :class:`~Sampler` that uses ITensor mps simulator for
    sampling.

    The following parameters including
    keyword arguments `**kwargs` are passed to `ITensors.apply
    <https://itensor.github.io/ITensors.jl/dev/MPSandMPO.html#ITensors.product-Tuple{ITensor,%20ITensors.AbstractMPS}>`_

    Args:
        maxdim: The maximum number of singular values.
        cutoff: Singular value truncation cutoff.
    """
    ensure_itensor_loaded()

    def sample(circuit: ImmutableQuantumCircuit, shots: int) -> MeasurementCounts:
        if maxdim is not None:
            kwargs["maxdim"] = maxdim
        if cutoff is not None:
            kwargs["cutoff"] = cutoff
        return _sample(circuit, shots, **kwargs)

    return sample


def _sample_concurrently(
    circuit_shots_tuples: Iterable[tuple[ImmutableQuantumCircuit, int]],
    executor: Optional["Executor"],
    concurrency: int = 1,
    **kwargs: Any,
) -> Iterable[MeasurementCounts]:
    def _sample_sequentially(
        _: Any, circuit_shots_tuples: Iterable[tuple[ImmutableQuantumCircuit, int]]
    ) -> Iterable[MeasurementCounts]:
        return [
            _sample(circuit, shots, **kwargs) for circuit, shots in circuit_shots_tuples
        ]

    return execute_concurrently(
        _sample_sequentially, None, circuit_shots_tuples, executor, concurrency
    )


def create_itensor_mps_concurrent_sampler(
    executor: Optional["Executor"] = None,
    concurrency: int = 1,
    *,
    maxdim: Optional[int] = None,
    cutoff: Optional[float] = None,
    **kwargs: Any,
) -> ConcurrentSampler:
    """Returns a :class:`~ConcurrentSampler` that uses ITensor mps simulator
    for sampling.

    The following parameters including
    keyword arguments `**kwargs` are passed to `ITensors.apply
    <https://itensor.github.io/ITensors.jl/dev/MPSandMPO.html#ITensors.product-Tuple{ITensor,%20ITensors.AbstractMPS}>`_

    Args:
        maxdim: The maximum number of singular values.
        cutoff: Singular value truncation cutoff.

    For now, this function works when the executor is defined like below

    Examples:
        >>> with ProcessPoolExecutor(
                max_workers=2, mp_context=get_context("spawn")
            ) as executor:
                sampler = create_itensor_mps_concurrent_sampler(
                    executor, 2, **kwargs
                )
                results = list(sampler([(circuit1, 1000), (circuit2, 2000)]))
    """
    ensure_itensor_loaded()

    if maxdim is not None:
        kwargs["maxdim"] = maxdim
    if cutoff is not None:
        kwargs["cutoff"] = cutoff

    def sampler(
        circuit_shots_tuples: Iterable[tuple[ImmutableQuantumCircuit, int]]
    ) -> Iterable[MeasurementCounts]:
        return _sample_concurrently(
            circuit_shots_tuples, executor, concurrency, **kwargs
        )

    return sampler
