# Licensed under the MIT License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      https://mit-license.org/
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from math import pi
from typing import Any, Final, Mapping, Optional, Sequence

from quri_parts.backend.units import TimeUnit, TimeValue
from quri_parts.circuit import ImmutableQuantumCircuit
from quri_parts.core.operator import PAULI_IDENTITY, Operator
from quri_parts.core.sampling import MeasurementCounts, Sampler
from quri_parts.core.sampling.default_sampler import DEFAULT_SAMPLER
from quri_parts.core.state import CircuitQuantumState
from quri_parts.core.utils.deprecation import deprecated_shots_kwarg
from quri_parts.qsub.compile import compile_sub
from quri_parts.qsub.eval import QURIPartsEvaluatorHooks
from quri_parts.qsub.evaluate import Evaluator
from quri_parts.qsub.machineinst import MachineSub
from quri_parts.qsub.op import Op, OpFactory
from quri_parts.qsub.primitive import AllBasicSet
from quri_parts.qsub.resolve import resolve_sub

from quri_algo.algo.interface import (
    AlgorithmResult,
    Analysis,
    Analyzer,
    AnalyzeResult,
    LoweringLevel,
    QuantumAlgorithm,
)
from quri_algo.algo.utils.mappings import CircuitMapping
from quri_algo.algo.utils.timer import timer
from quri_algo.circuit_lib.qpe import QPE as QPEop
from quri_algo.circuit_lib.time_evolution.interface import TimeEvolutionOpFactory
from quri_algo.core.estimator.hadamard_test import remap_state_for_hadamard_test
from quri_algo.problem import QubitHamiltonian


def get_ancilla_register_counts(
    counts: MeasurementCounts, ancilla_count: int
) -> Mapping[int, int]:
    """Collapse raw measurement counts onto the ancilla register by masking out
    the trial-state qubits and summing counts that share the same ancilla
    bitstring.

    Args:
        counts: Measurement counts over the full circuit (ancilla and trial
            state qubits).
        ancilla_count: Number of ancilla qubits, assumed to occupy the
            lowest-index bits.

    Returns:
        Mapping from ancilla register value to its summed count.

    Raises:
        ValueError: If ``counts`` holds float values, which indicates an
            ideal (non-shot-based) sampler was used.
    """
    bitmask = (0b1 << ancilla_count) - 1
    ancilla_register_counts: dict[int, int] = {}
    for b, c in counts.items():
        if isinstance(c, float):
            raise ValueError("Expected counts to be int. Did you use an ideal sampler?")
        k = b & bitmask
        if ancilla_register_counts.get(k):
            ancilla_register_counts[k] += c
        else:
            ancilla_register_counts[k] = c

    return ancilla_register_counts


def qpe_counts_to_eigenvalues(
    ancilla_register_counts: MeasurementCounts,
    ancilla_count: int,
    scale_factor: float = 1.0,
    constant: float = 0.0,
) -> Mapping[int, float]:
    """Convert measured ancilla register values into estimated eigenvalues.

    Args:
        ancilla_register_counts: Counts keyed by ancilla register value, as
            returned by :func:`get_ancilla_register_counts`.
        ancilla_count: Number of ancilla qubits.
        scale_factor: Factor the phase was scaled by before encoding (e.g.
            the QPE evolution time), used to undo that scaling.
        constant: Constant term to add back after rescaling (e.g. a
            Hamiltonian's identity term that was removed before QPE).

    Returns:
        Mapping from ancilla register value to its estimated eigenvalue.
    """
    eigen_values = {
        k: k / (scale_factor * 2**ancilla_count) + constant
        for k in ancilla_register_counts.keys()
    }

    return eigen_values


def get_qpe_msub(
    ancilla_count: int,
    hamiltonian: QubitHamiltonian,
    time_evo_op_factory: TimeEvolutionOpFactory,
    scale_factor: float,
    *args: Any,
    primitives: Sequence[Op | OpFactory[float]] = AllBasicSet,
) -> MachineSub:
    """Build the compiled machine-sub-circuit that implements QPE for the given
    Hamiltonian.

    Args:
        ancilla_count: Number of ancilla qubits used for phase readout.
        hamiltonian: Hamiltonian whose phase (eigenvalue) is estimated.
        time_evo_op_factory: Factory producing the time-evolution operator
            for the Hamiltonian.
        scale_factor: Evolution time (or other scaling) applied to the
            Hamiltonian before time evolution.
        *args: Extra positional arguments forwarded to
            ``time_evo_op_factory``.
        primitives: Gate set QPE's sub-circuit is compiled down to.

    Returns:
        Compiled MachineSub implementing QPE.
    """
    time_evolution_op = time_evo_op_factory(hamiltonian, -2 * pi * scale_factor, *args)
    qpe_sub = resolve_sub(QPEop(ancilla_count, time_evolution_op))
    assert qpe_sub is not None
    qpe_msub = compile_sub(qpe_sub, primitives)
    return qpe_msub


def remove_constant_term(h: QubitHamiltonian) -> QubitHamiltonian:
    """Return a copy of the Hamiltonian with its identity (constant) term
    dropped.

    Args:
        h: Hamiltonian to strip the constant term from.

    Returns:
        QubitHamiltonian with the same non-identity terms as ``h``, and no
        identity term.
    """
    op = Operator()
    for p, c in h.qubit_hamiltonian.items():
        if p != PAULI_IDENTITY:
            op.add_term(p, c)
    return QubitHamiltonian(h.n_qubit, op)


class QPEResult(AlgorithmResult):
    """QPE result."""

    def __init__(
        self,
        algorithm: QuantumAlgorithm,
        eigen_values: Mapping[int, float],
        ancilla_register_counts: Mapping[int, int],
        elapsed_time: Optional[float] = None,
    ):
        """Initialize the result with the estimated eigenvalues and their
        supporting ancilla register counts.

        Args:
            algorithm: QPE algorithm instance that produced this result.
            eigen_values: Estimated eigenvalue for each measured ancilla
                register value.
            ancilla_register_counts: Measurement counts keyed by ancilla
                register value.
            elapsed_time: Wall-clock time the run took, in seconds, if
                measured.
        """
        AlgorithmResult.__init__(self, algorithm, elapsed_time)
        self.eigen_values = eigen_values
        self.ancilla_register_counts = ancilla_register_counts


class QPEAnalysis(Analysis):
    """QPE resource analysis."""

    def __init__(
        self,
        lowering_level: LoweringLevel,
        circuit_gate_count: Mapping[ImmutableQuantumCircuit, int],
        circuit_depth: Mapping[ImmutableQuantumCircuit, int],
        circuit_latency: Mapping[ImmutableQuantumCircuit, TimeValue | None],
        circuit_execution_count: Mapping[ImmutableQuantumCircuit, int],
        circuit_fidelities: Mapping[ImmutableQuantumCircuit, float | None],
        circuit_qubit_count: Mapping[ImmutableQuantumCircuit, int],
        circuit_physical_qubit_count: Mapping[ImmutableQuantumCircuit, int],
    ) -> None:
        """Initialize the analysis from its per-circuit metrics; see the base
        class docstring for what each mapping holds."""
        super().__init__(
            lowering_level,
            circuit_gate_count,
            circuit_depth,
            circuit_latency,
            circuit_execution_count,
            circuit_fidelities,
            circuit_qubit_count,
            circuit_physical_qubit_count,
        )

    @property
    def total_latency(self) -> TimeValue:
        """Total latency of the circuit."""
        assert len(self.circuit_latency) == 1
        latency = list(self.circuit_latency.values())[0]
        assert latency
        assert len(self.circuit_execution_count) == 1
        latency_in_ns = latency.in_ns() * list(self.circuit_execution_count.values())[0]
        return TimeValue(latency_in_ns, TimeUnit.NANOSECOND)

    @property
    def max_physical_qubit_count(self) -> int:
        """Total number of physical qubits of the circuit."""
        assert len(self.circuit_physical_qubit_count) == 1
        return list(self.circuit_physical_qubit_count.values())[0]

    @classmethod
    def from_circuit_list(
        cls,
        circuit_analysis: AnalyzeResult,
        total_circuit_executions: int,
        circuit_list: Sequence[ImmutableQuantumCircuit],
    ) -> "QPEAnalysis":
        """Build a QPEAnalysis by applying a single circuit analysis to every
        circuit in the list.

        Args:
            circuit_analysis: AnalyzeResult shared by all circuits (QPE
                runs a single circuit, so this is the analysis of that
                circuit).
            total_circuit_executions: Number of times the circuit is
                executed.
            circuit_list: Circuits to build the analysis for.

        Returns:
            The resulting QPEAnalysis.
        """
        return cls(
            lowering_level=circuit_analysis.lowering_level,
            circuit_depth=CircuitMapping(
                [(c, circuit_analysis.depth) for c in circuit_list]
            ),
            circuit_gate_count=CircuitMapping(
                [(c, circuit_analysis.gate_count) for c in circuit_list]
            ),
            circuit_latency=CircuitMapping(
                [(c, circuit_analysis.latency) for c in circuit_list]
            ),
            circuit_execution_count=CircuitMapping(
                [(c, total_circuit_executions) for c in circuit_list]
            ),
            circuit_fidelities=CircuitMapping(
                [(c, circuit_analysis.fidelity) for c in circuit_list]
            ),
            circuit_qubit_count=CircuitMapping(
                [(c, circuit_analysis.qubit_count) for c in circuit_list]
            ),
            circuit_physical_qubit_count=CircuitMapping(
                [(c, circuit_analysis.physical_qubit_count) for c in circuit_list]
            ),
        )


@deprecated_shots_kwarg
def execute_qpe(
    algorithm_instance: QuantumAlgorithm,
    trial_state: CircuitQuantumState,
    sampler: Sampler,
    shots: int,
    ancilla_count: int,
    qpe_msub: MachineSub,
    **kwargs: Any,
) -> QPEResult:
    """Run QPE by sampling the given machine-sub-circuit and converting the
    measured ancilla register into eigenvalue estimates.

    Args:
        algorithm_instance: QPE algorithm instance running this execution.
        trial_state: State QPE estimates the eigenvalue of.
        sampler: Sampler used to measure the composed circuit.
        shots: Number of sampling shots.
        ancilla_count: Number of ancilla qubits used for phase readout.
        qpe_msub: Compiled QPE machine-sub-circuit, as built by
            :func:`get_qpe_msub`.
        **kwargs: Extra keyword arguments forwarded to
            :func:`qpe_counts_to_eigenvalues` (e.g. ``scale_factor``,
            ``constant``).

    Returns:
        QPEResult holding the estimated eigenvalues and ancilla counts.
    """
    qp_generator = Evaluator(QURIPartsEvaluatorHooks())
    qp_circuit: ImmutableQuantumCircuit = qp_generator.run(qpe_msub).freeze()
    padded_trial_state = remap_state_for_hadamard_test(trial_state, ancilla_count)
    total_circuit = padded_trial_state.circuit + qp_circuit
    counts = dict(sampler(total_circuit, shots))
    ancilla_register_counts = get_ancilla_register_counts(counts, ancilla_count)
    eigen_values = qpe_counts_to_eigenvalues(
        ancilla_register_counts, ancilla_count, **kwargs
    )

    return QPEResult(algorithm_instance, eigen_values, ancilla_register_counts)


@deprecated_shots_kwarg
def analyze_qpe(
    trial_state: CircuitQuantumState,
    shots: int,
    ancilla_count: int,
    analyzer: Analyzer,
    qpe_msub: MachineSub,
) -> QPEAnalysis:
    """Analyze the quantum resources QPE would need for the given machine-sub-
    circuit, without executing it.

    Args:
        trial_state: State QPE would estimate the eigenvalue of.
        shots: Number of sampling shots QPE would use.
        ancilla_count: Number of ancilla qubits used for phase readout.
        analyzer: Analyzer used to estimate the composed circuit's cost.
        qpe_msub: Compiled QPE machine-sub-circuit, as built by
            :func:`get_qpe_msub`.

    Returns:
        QPEAnalysis holding the resource estimates.
    """
    qp_generator = Evaluator(QURIPartsEvaluatorHooks())
    qp_circuit: ImmutableQuantumCircuit = qp_generator.run(qpe_msub).freeze()
    padded_trial_state = remap_state_for_hadamard_test(trial_state, ancilla_count)
    total_circuit = padded_trial_state.circuit + qp_circuit
    analysis = analyzer(total_circuit)

    return QPEAnalysis.from_circuit_list(analysis, shots, [qp_circuit])


class QPE(QuantumAlgorithm):
    """Quantum Phase Estimation (QPE)

    This class provides an implementation of QPE.
    """

    _name: Final[str] = "Quantum Phase Estimation (QPE)"

    def __init__(
        self,
        time_evolution_op_factory: TimeEvolutionOpFactory,
        sampler: Sampler = DEFAULT_SAMPLER,
    ):
        """Initialize QPE with the time-evolution operator factory and sampler
        it will use.

        Args:
            time_evolution_op_factory: Factory producing the time-evolution
                operator for a given Hamiltonian, used to build the QPE
                sub-circuit.
            sampler: Sampler used to measure QPE circuits.
        """
        self._time_evolution_op_factory = time_evolution_op_factory
        self._sampler = sampler

    @property
    def name(self) -> str:
        """Name of the algorithm."""
        return self._name

    @property
    def time_evolution_op_factory(self) -> TimeEvolutionOpFactory:
        """Factory producing the time-evolution operator used to build the QPE
        sub-circuit."""
        return self._time_evolution_op_factory

    @property
    def sampler(self) -> Sampler:
        """Sampler used to measure QPE circuits."""
        return self._sampler

    @deprecated_shots_kwarg
    def run(
        self,
        trial_state: CircuitQuantumState,
        shots: int,
        ancilla_count: int,
        qpe_msub: MachineSub,
        **kwargs: Any,
    ) -> QPEResult:
        """Run QPE on the given machine-sub-circuit; see
        :func:`execute_qpe` for the parameters and return value."""
        return execute_qpe(
            self, trial_state, self.sampler, shots, ancilla_count, qpe_msub, **kwargs
        )

    @deprecated_shots_kwarg
    def analyze(
        self,
        trial_state: CircuitQuantumState,
        shots: int,
        ancilla_count: int,
        analyzer: Analyzer,
        qpe_msub: MachineSub,
    ) -> QPEAnalysis:
        """Analyze QPE's resource requirements for the given machine-sub-
        circuit; see :func:`analyze_qpe` for the parameters and return
        value."""
        return analyze_qpe(trial_state, shots, ancilla_count, analyzer, qpe_msub)


class TimeEvolutionQPE(QuantumAlgorithm):
    """Time-evolution Quantum Phase Estimation (QPE)

    This class provides an implementation of QPE using Hamiltonian time
    evolution.
    """

    _name: Final[str] = "Time-evolution Quantum Phase Estimation (QPE)"

    def __init__(
        self,
        time_evolution_op_factory: TimeEvolutionOpFactory,
        sampler: Sampler = DEFAULT_SAMPLER,
    ):
        """Initialize QPE with the time-evolution operator factory and sampler
        it will use.

        Args:
            time_evolution_op_factory: Factory producing the time-evolution
                operator for a given Hamiltonian, used to build the QPE
                sub-circuit.
            sampler: Sampler used to measure QPE circuits.
        """
        self._time_evolution_op_factory = time_evolution_op_factory
        self._sampler = sampler

    @property
    def name(self) -> str:
        """Name of the algorithm."""
        return self._name

    @property
    def time_evolution_op_factory(self) -> TimeEvolutionOpFactory:
        """Factory producing the time-evolution operator used to build the QPE
        sub-circuit."""
        return self._time_evolution_op_factory

    @property
    def sampler(self) -> Sampler:
        """Sampler used to measure QPE circuits."""
        return self._sampler

    @deprecated_shots_kwarg
    @timer
    def run(
        self,
        trial_state: CircuitQuantumState,
        shots: int,
        ancilla_count: int,
        hamiltonian: QubitHamiltonian,
        evolution_time: float,
        *args: Any,
        primitives: Sequence[Op | OpFactory[float]] = AllBasicSet,
    ) -> QPEResult:
        """Run QPE to estimate an eigenvalue of the given Hamiltonian via time
        evolution.

        Args:
            trial_state: State QPE estimates the eigenvalue of.
            shots: Number of sampling shots.
            ancilla_count: Number of ancilla qubits used for phase readout.
            hamiltonian: Hamiltonian whose phase (eigenvalue) is estimated.
            evolution_time: Time evolution is applied for; also used to
                rescale the measured phase back into an eigenvalue.
            *args: Extra positional arguments forwarded to the
                time-evolution operator factory.
            primitives: Gate set QPE's sub-circuit is compiled down to.

        Returns:
            QPEResult holding the estimated eigenvalues and ancilla counts.
        """
        constant = hamiltonian.qubit_hamiltonian.constant
        if constant:
            hamiltonian = remove_constant_term(hamiltonian)
        qpe_msub = get_qpe_msub(
            ancilla_count,
            hamiltonian,
            self.time_evolution_op_factory,
            evolution_time,
            *args,
            primitives=primitives,
        )

        return execute_qpe(
            self,
            trial_state,
            self.sampler,
            shots,
            ancilla_count,
            qpe_msub,
            scale_factor=evolution_time,
            constant=constant,
        )

    @deprecated_shots_kwarg
    def analyze(
        self,
        trial_state: CircuitQuantumState,
        shots: int,
        ancilla_count: int,
        analyzer: Analyzer,
        hamiltonian: QubitHamiltonian,
        evolution_time: float,
        *args: Any,
        primitives: Sequence[Op | OpFactory[float]] = AllBasicSet,
    ) -> QPEAnalysis:
        """Analyze the quantum resources this time-evolution QPE run would
        need, without executing it.

        Args:
            trial_state: State QPE would estimate the eigenvalue of.
            shots: Number of sampling shots QPE would use.
            ancilla_count: Number of ancilla qubits used for phase readout.
            analyzer: Analyzer used to estimate the composed circuit's
                cost.
            hamiltonian: Hamiltonian whose phase (eigenvalue) is estimated.
            evolution_time: Time evolution would be applied for.
            *args: Extra positional arguments forwarded to the
                time-evolution operator factory.
            primitives: Gate set QPE's sub-circuit is compiled down to.

        Returns:
            QPEAnalysis holding the resource estimates.
        """
        qpe_msub = get_qpe_msub(
            ancilla_count,
            hamiltonian,
            self.time_evolution_op_factory,
            evolution_time,
            *args,
            primitives=primitives,
        )

        return analyze_qpe(trial_state, shots, ancilla_count, analyzer, qpe_msub)
