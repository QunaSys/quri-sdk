from math import pi
from typing import Any, Final, Mapping, Optional, Sequence

from quri_parts.backend.units import TimeUnit, TimeValue
from quri_parts.circuit import ImmutableQuantumCircuit
from quri_parts.core.operator import PAULI_IDENTITY, Operator
from quri_parts.core.sampling import MeasurementCounts, Sampler
from quri_parts.core.sampling.default_sampler import DEFAULT_SAMPLER
from quri_parts.core.state import CircuitQuantumState
from quri_parts.qsub.compile import compile_sub
from quri_parts.qsub.eval import QURIPartsEvaluatorHooks
from quri_parts.qsub.evaluate import Evaluator
from quri_parts.qsub.lib.qpe import QPE as QPEop
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
from quri_algo.core.estimator.hadamard_test import remap_state_for_hadamard_test
from quri_algo.problem import QubitHamiltonian
from quri_algo.qsub.time_evolution.interface import TimeEvolutionOpFactory


def get_ancilla_register_counts(
    counts: MeasurementCounts, ancilla_count: int
) -> Mapping[int, int]:
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
    time_evolution_op = time_evo_op_factory(hamiltonian, -2 * pi * scale_factor, *args)
    qpe_sub = resolve_sub(QPEop(ancilla_count, time_evolution_op))
    assert qpe_sub is not None
    qpe_msub = compile_sub(qpe_sub, primitives)
    return qpe_msub


def remove_constant_term(h: QubitHamiltonian) -> QubitHamiltonian:
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
    ) -> None:
        super().__init__(
            lowering_level,
            circuit_gate_count,
            circuit_depth,
            circuit_latency,
            circuit_execution_count,
            circuit_fidelities,
            circuit_qubit_count,
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
        assert len(self.circuit_qubit_count) == 1
        return list(self.circuit_qubit_count.values())[0]

    @classmethod
    def from_circuit_list(
        cls,
        circuit_analysis: AnalyzeResult,
        total_circuit_executions: int,
        circuit_list: Sequence[ImmutableQuantumCircuit],
    ) -> "QPEAnalysis":
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
        )


def execute_qpe(
    algorithm_instance: QuantumAlgorithm,
    trial_state: CircuitQuantumState,
    sampler: Sampler,
    n_shots: int,
    ancilla_count: int,
    qpe_msub: MachineSub,
    **kwargs: Any,
) -> QPEResult:
    qp_generator = Evaluator(QURIPartsEvaluatorHooks())
    qp_circuit: ImmutableQuantumCircuit = qp_generator.run(qpe_msub).freeze()
    padded_trial_state = remap_state_for_hadamard_test(trial_state, ancilla_count)
    total_circuit = padded_trial_state.circuit + qp_circuit
    counts = dict(sampler(total_circuit, n_shots))
    ancilla_register_counts = get_ancilla_register_counts(counts, ancilla_count)
    eigen_values = qpe_counts_to_eigenvalues(
        ancilla_register_counts, ancilla_count, **kwargs
    )

    return QPEResult(algorithm_instance, eigen_values, ancilla_register_counts)


def analyze_qpe(
    trial_state: CircuitQuantumState,
    n_shots: int,
    ancilla_count: int,
    analyzer: Analyzer,
    qpe_msub: MachineSub,
) -> QPEAnalysis:
    qp_generator = Evaluator(QURIPartsEvaluatorHooks())
    qp_circuit: ImmutableQuantumCircuit = qp_generator.run(qpe_msub).freeze()
    padded_trial_state = remap_state_for_hadamard_test(trial_state, ancilla_count)
    total_circuit = padded_trial_state.circuit + qp_circuit
    analysis = analyzer(total_circuit)

    return QPEAnalysis.from_circuit_list(analysis, n_shots, [qp_circuit])


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
        self._time_evolution_op_factory = time_evolution_op_factory
        self._sampler = sampler

    @property
    def name(self) -> str:
        return self._name

    @property
    def time_evolution_op_factory(self) -> TimeEvolutionOpFactory:
        return self._time_evolution_op_factory

    @property
    def sampler(self) -> Sampler:
        return self._sampler

    def run(
        self,
        trial_state: CircuitQuantumState,
        n_shots: int,
        ancilla_count: int,
        qpe_msub: MachineSub,
        **kwargs: Any,
    ) -> QPEResult:
        return execute_qpe(
            self, trial_state, self.sampler, n_shots, ancilla_count, qpe_msub, **kwargs
        )

    def analyze(
        self,
        trial_state: CircuitQuantumState,
        n_shots: int,
        ancilla_count: int,
        analyzer: Analyzer,
        qpe_msub: MachineSub,
    ) -> QPEAnalysis:
        return analyze_qpe(trial_state, n_shots, ancilla_count, analyzer, qpe_msub)


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
        self._time_evolution_op_factory = time_evolution_op_factory
        self._sampler = sampler

    @property
    def name(self) -> str:
        return self._name

    @property
    def time_evolution_op_factory(self) -> TimeEvolutionOpFactory:
        return self._time_evolution_op_factory

    @property
    def sampler(self) -> Sampler:
        return self._sampler

    @timer
    def run(
        self,
        trial_state: CircuitQuantumState,
        n_shots: int,
        ancilla_count: int,
        hamiltonian: QubitHamiltonian,
        evolution_time: float,
        *args: Any,
        primitives: Sequence[Op | OpFactory[float]] = AllBasicSet,
    ) -> QPEResult:
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
            n_shots,
            ancilla_count,
            qpe_msub,
            scale_factor=evolution_time,
            constant=constant,
        )

    def analyze(
        self,
        trial_state: CircuitQuantumState,
        n_shots: int,
        ancilla_count: int,
        analyzer: Analyzer,
        hamiltonian: QubitHamiltonian,
        evolution_time: float,
        *args: Any,
        primitives: Sequence[Op | OpFactory[float]] = AllBasicSet,
    ) -> QPEAnalysis:
        qpe_msub = get_qpe_msub(
            ancilla_count,
            hamiltonian,
            self.time_evolution_op_factory,
            evolution_time,
            *args,
            primitives=primitives,
        )

        return analyze_qpe(trial_state, n_shots, ancilla_count, analyzer, qpe_msub)
