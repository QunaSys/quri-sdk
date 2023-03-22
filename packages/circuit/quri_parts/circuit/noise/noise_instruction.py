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
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import MutableMapping, MutableSequence, Sequence
from dataclasses import dataclass
from typing import Protocol, Union, cast

import numpy as np
import numpy.linalg as la
import numpy.typing as npt
from typing_extensions import TypeAlias

from quri_parts.circuit import NonParametricQuantumCircuit, QuantumGate
from quri_parts.circuit.gate_names import NonParametricGateNameType

#: Represents a backend-independent noise instruction to be added to NoiseModel.
NoiseInstruction: TypeAlias = Union["CircuitNoiseInstruction", "GateNoiseInstruction"]
KrausOperatorSequence: TypeAlias = tuple[npt.NDArray[np.float64], ...]
GateMatrices: TypeAlias = tuple[npt.NDArray[np.float64], ...]

QubitIndices: TypeAlias = Sequence[int]
QubitNoisePair: TypeAlias = tuple[QubitIndices, "GateNoiseInstruction"]


def _check_valid_probability(x: float, name: str) -> None:
    """Probability argument checker.

    Checks whether ``x`` is a correct value as a probability (i.e.,
    greater than or equal to 0 and less than or equal to 1), and if
    wrong, raises a ValueError with a message including the variable
    ``name``.
    """
    if x < 0 or x > 1:
        raise ValueError(f"{name} must be between 0 and 1 (inclusive) but it was {x}.")


class CircuitNoiseInstruction(ABC):
    """Represents the noise applied depending on the structure of the
    circuit."""

    def __init__(self, name: str):
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @abstractmethod
    def create_resolver(self) -> "CircuitNoiseResolverProtocol":
        """Returns new :class:`CircuitNoiseResolverProtocol` instance for each
        concrete class of :class:`CircuitNoiseInstruction`.

        When converting from the original circuit to a circuit with the
        noise model applied for each backend, ``noises_for_gate()``
        method and ``noises_for_depth()`` method of
        :class:`CircuitNoiseResolverProtocol` are called for each gate
        while scanning the circuit from front to back, returning the
        noises that needs to be applied to each position in the circuit.
        """
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name})"


class CircuitNoiseResolverProtocol(Protocol):
    """This is an helper object used when a quantum circuit is converted into a
    concrete backend circuit."""

    def noises_for_gate(
        self, gate: QuantumGate, index: int, circuit: NonParametricQuantumCircuit
    ) -> Sequence[QubitNoisePair]:
        """Returns the noises that should be inserted at the ``index`` position
        in the original circuit by being called with the ``gate`` from the
        front while scanning the circuit.

        Each :class:`CircuitNoiseInstruction` must implement its own
        :class:`TraversalIndicatorProtocol` and must returns the instance
        by ``create_traversal_indicator()`` method.

        Args:
            gate: :class:`QuantumGate` at `index` position.
            index: An index indicating the position of the `gate` in the circuit.
            circuit: Reference of the target circuit.
        """
        return []

    def noises_for_depth(
        self, qubit: int, depths: Sequence[int], circuit: NonParametricQuantumCircuit
    ) -> Sequence[QubitNoisePair]:
        """Returns qubit and noise pairs that shoud be applied to the ``qubit``
        within ``depths``.

        For each qubit, it is assumed that this function will be called each time
        the depth is progressed by each gate or the scan reaches to the end of the
        circuit while scanning from the front.

        Args:
            qubit: Target qubit indice.
            depths: Sequence of progressed depths of ``qubit``.
            circuit: Reference of the target circuit.
        """
        return []


@dataclass(frozen=True)
class GateNoiseInstruction:
    """Represents the noise that is applied when individual gates act on
    qubits."""

    name: str
    qubit_count: int
    params: tuple[float, ...]
    qubit_indices: tuple[int, ...]
    target_gates: tuple[NonParametricGateNameType, ...]


class GateIntervalNoise(CircuitNoiseInstruction):
    """For each qubit, given single qubit noises are applied each time a
    certain number of gates are applied.

    Args:
        single_qubit_noises: Sequence of single qubit :class:`GateNoiseInstruction`.
        gate_interval: Gate interval. For each qubit, ``single_qubit_noises`` are
            applied every time ``gate_interval`` gates are applied.
    """

    def __init__(
        self,
        single_qubit_noises: Sequence[GateNoiseInstruction],
        gate_interval: int,
    ) -> None:
        if not gate_interval > 0:
            raise ValueError("gate_interval must be greater than 0.")
        if not all(noise.qubit_count == 1 for noise in single_qubit_noises):
            raise ValueError("single_qubit_noises cannot contain multi qubit noises.")
        self._gate_interval = gate_interval
        self._single_qubit_noises = tuple(single_qubit_noises)
        super().__init__("GateIntervalNoise")

    @property
    def gate_interval(self) -> int:
        return self._gate_interval

    @property
    def single_qubit_noises(self) -> Sequence[GateNoiseInstruction]:
        return self._single_qubit_noises

    class GateIntervalResolver(CircuitNoiseResolverProtocol):
        def __init__(self, gate_interval: int, noises: Sequence[GateNoiseInstruction]):
            self._gate_interval: int = gate_interval
            self._noises: Sequence[GateNoiseInstruction] = noises
            self._qubit_indicators: MutableMapping[int, int] = defaultdict(int)

        def noises_for_gate(
            self,
            gate: QuantumGate,
            index: int,
            circuit: NonParametricQuantumCircuit,
        ) -> Sequence[QubitNoisePair]:
            ret = []
            for q in tuple(gate.control_indices) + tuple(gate.target_indices):
                self._qubit_indicators[q] += 1
                if self._qubit_indicators[q] >= self._gate_interval:
                    self._qubit_indicators[q] = 0
                    ret.extend([((q,), n) for n in self._noises])
            return ret

    def create_resolver(self) -> CircuitNoiseResolverProtocol:
        return self.GateIntervalResolver(self.gate_interval, self.single_qubit_noises)


class DepthIntervalNoise(CircuitNoiseInstruction):
    """Trace the gates of the :class:`QuantumCircuit` from the front to the
    back, and apply the given single qubit :class:`GateNoiseInstruction` to all
    qubits every time a certain depth is advanced.

    Args:
        single_qubit_noises: Sequence of single qubit :class:`GateNoiseInstruction`.
        depth_interval: Depth interval. Every time the depth advances
            ``depth_interval``, ``single_qubit_noises`` are applied to all qubits.
    """

    def __init__(
        self,
        single_qubit_noises: Sequence[GateNoiseInstruction],
        depth_interval: int,
    ) -> None:
        if not depth_interval > 0:
            raise ValueError("depth_interval must be greater than 0.")
        if not all(noise.qubit_count == 1 for noise in single_qubit_noises):
            raise ValueError("single_qubit_noises cannot contain multi qubit noises.")
        self._depth_interval = depth_interval
        self._single_qubit_noises = tuple(single_qubit_noises)
        super().__init__("DepthIntervalNoise")

    @property
    def depth_interval(self) -> int:
        return self._depth_interval

    @property
    def single_qubit_noises(self) -> Sequence[GateNoiseInstruction]:
        return self._single_qubit_noises

    class DepthIntervalResolver(CircuitNoiseResolverProtocol):
        def __init__(self, depth_interval: int, noises: Sequence[GateNoiseInstruction]):
            self._depth_interval: int = depth_interval
            self._noises: Sequence[GateNoiseInstruction] = noises

        def noises_for_depth(
            self,
            qubit: int,
            depths: Sequence[int],
            circuit: NonParametricQuantumCircuit,
        ) -> Sequence[QubitNoisePair]:
            ret: MutableSequence[QubitNoisePair] = []
            pairs = [((qubit,), n) for n in self._noises]
            for d in depths:
                if d > 0 and d % self._depth_interval == 0:
                    ret.extend(pairs)
            return ret

    def create_resolver(self) -> CircuitNoiseResolverProtocol:
        return self.DepthIntervalResolver(self.depth_interval, self.single_qubit_noises)


class MeasurementNoise(CircuitNoiseInstruction):
    """Represents the noise which occurs during the measurement of a qubit. At
    the right end of the circuit, it applies single qubit noises to all qubits
    or specified qubits (if given).

    Args:
        single_qubit_noises: Sequence of single qubit :class:`GateNoiseInstruction`.
        qubit_indices: Sequence of target qubit indices. If empty, all qubits will
            be covered.
    """

    def __init__(
        self,
        single_qubit_noises: Sequence[GateNoiseInstruction],
        qubit_indices: Sequence[int] = [],
    ):
        self._qubit_indices = tuple(qubit_indices)
        self._single_qubit_noises = single_qubit_noises
        super().__init__("MeasurementNoise")

    @property
    def qubit_indices(self) -> Sequence[int]:
        return self._qubit_indices

    @property
    def apply_to_all_qubits(self) -> bool:
        return not self._qubit_indices

    @property
    def single_qubit_noises(self) -> Sequence[GateNoiseInstruction]:
        return self._single_qubit_noises

    class MeasurementResolver(CircuitNoiseResolverProtocol):
        def __init__(
            self, qubits: Sequence[int], noises: Sequence[GateNoiseInstruction]
        ):
            self._qubits: Sequence[int] = qubits
            self._noises: Sequence[GateNoiseInstruction] = noises

        def noises_for_depth(
            self,
            qubit: int,
            depths: Sequence[int],
            circuit: NonParametricQuantumCircuit,
        ) -> Sequence[QubitNoisePair]:
            if circuit.depth in depths:
                if not self._qubits or qubit in self._qubits:
                    return [((qubit,), n) for n in self._noises]
            return []

    def create_resolver(self) -> CircuitNoiseResolverProtocol:
        return self.MeasurementResolver(self.qubit_indices, self.single_qubit_noises)


class BitFlipNoise(GateNoiseInstruction):
    """Single qubit bit flip noise.

    Args:
        error_prob: Probability of noise generation.
        qubit_indices: Sequence of target qubit indices.
        target_gates: Sequence of target gate names.
    """

    def __init__(
        self,
        error_prob: float,
        qubit_indices: Sequence[int] = (),
        target_gates: Sequence[NonParametricGateNameType] = (),
    ):
        _check_valid_probability(error_prob, "error_prob")
        super().__init__(
            name="BitFlipNoise",
            qubit_count=1,
            params=(error_prob,),
            qubit_indices=tuple(qubit_indices),
            target_gates=tuple(target_gates),
        )

    @property
    def error_prob(self) -> float:
        return self.params[0]


class PhaseFlipNoise(GateNoiseInstruction):
    """Single qubit phase flip noise.

    Args:
        error_prob: Probability of noise generation.
        qubit_indices: Sequence of target qubit indices.
        target_gates: Sequence of target gate names.
    """

    def __init__(
        self,
        error_prob: float,
        qubit_indices: Sequence[int] = (),
        target_gates: Sequence[NonParametricGateNameType] = (),
    ):
        _check_valid_probability(error_prob, "error_prob")
        super().__init__(
            name="PhaseFlipNoise",
            qubit_count=1,
            params=(error_prob,),
            qubit_indices=tuple(qubit_indices),
            target_gates=tuple(target_gates),
        )

    @property
    def error_prob(self) -> float:
        return self.params[0]


class BitPhaseFlipNoise(GateNoiseInstruction):
    """Single qubit bit and phase flip noise.

    Args:
        error_prob: Probability of noise generation.
        qubit_indices: Sequence of target qubit indices.
        target_gates: Sequence of target gate names.
    """

    def __init__(
        self,
        error_prob: float,
        qubit_indices: Sequence[int] = (),
        target_gates: Sequence[NonParametricGateNameType] = (),
    ):
        _check_valid_probability(error_prob, "error_prob")
        super().__init__(
            name="BitPhaseFlipNoise",
            qubit_count=1,
            params=(error_prob,),
            qubit_indices=tuple(qubit_indices),
            target_gates=tuple(target_gates),
        )

    @property
    def error_prob(self) -> float:
        return self.params[0]


class DepolarizingNoise(GateNoiseInstruction):
    """Single qubit depolarizing noise.

    Args:
        error_prob: Probability of noise generation.
        qubit_indices: Sequence of target qubit indices.
        target_gates: Sequence of target gate names.
    """

    def __init__(
        self,
        error_prob: float,
        qubit_indices: Sequence[int] = (),
        target_gates: Sequence[NonParametricGateNameType] = (),
    ):
        _check_valid_probability(error_prob, "error_prob")
        super().__init__(
            name="DepolarizingNoise",
            qubit_count=1,
            params=(error_prob,),
            qubit_indices=tuple(qubit_indices),
            target_gates=tuple(target_gates),
        )

    @property
    def error_prob(self) -> float:
        return self.params[0]


def _check_valid_qubit_indices(qubit_count: int, qubit_indices: Sequence[int]) -> None:
    """Check if ``qubit_indices`` is in the correct form of ``qubit_count``
    qubit noise."""
    if qubit_indices and qubit_count > 1 and qubit_count != len(qubit_indices):
        raise ValueError(
            "For multi qubit noise, qubit_indices must be either an empty list"
            "(accepts any qubits) or a list of length ``qubit_count`` (exact match)."
            "For single qubit noise, qubit_indices must be either an empty list"
            "(accepts any qubits) or a list of any length (match any of these)."
        )


def _is_aligned_square_matrices(xss: Sequence[Sequence[Sequence[float]]]) -> bool:
    """Tests that the elements of a given sequence are all square matrices of
    the same shape."""
    ss: npt.NDArray[np.int64] = np.array([np.array(x).shape for x in xss])
    return ss[0, 0] >= 2 and ss[0, 0] == ss[0, 1] and cast(bool, np.all(ss == ss[0]))


class PauliNoise(GateNoiseInstruction):
    """Multi qubit Pauli noise.

    In the case of multi qubit noise, qubit_indices should specify the qubits on
    which the target gate is acting, in any order, without excess or deficiency.

    Args:
        pauli_list: Sequence of series of Pauli ids.
        prob_list: Sequence of probability for each Pauli operations.
        qubit_indices: Sequence of target qubit indices.
        target_gates: Sequence of target gate names.
        eq_tolerance: Allowed error in the total probability over 1.
    """

    def __init__(
        self,
        pauli_list: Sequence[Sequence[int]],
        prob_list: Sequence[float],
        qubit_indices: Sequence[int] = (),
        target_gates: Sequence[NonParametricGateNameType] = (),
        eq_tolerance: float = 1.0e-8,
    ):
        if not pauli_list:
            raise ValueError("pauli_list cannot be empty.")
        if not prob_list:
            raise ValueError("prob_list cannot be empty.")
        if len(pauli_list) != len(prob_list):
            raise ValueError("pauli_list and prob_list must have the same length.")
        for i, prob in enumerate(prob_list):
            _check_valid_probability(prob, f"prob_list[{i}]")
        if sum(prob_list) - 1.0 > eq_tolerance:
            raise ValueError("The sum of prob_list must be less than or equal to 1.")
        self._pauli_list = tuple(pauli_list)
        self._prob_list = tuple(prob_list)
        qubit_count = self._get_qubit_count()
        _check_valid_qubit_indices(qubit_count, qubit_indices)
        super().__init__(
            name="PauliNoise",
            qubit_count=qubit_count,
            params=(),
            qubit_indices=tuple(qubit_indices),
            target_gates=tuple(target_gates),
        )

    @property
    def pauli_list(self) -> Sequence[Sequence[int]]:
        return self._pauli_list

    @property
    def prob_list(self) -> Sequence[float]:
        return self._prob_list

    def _get_qubit_count(self) -> int:
        if np.setdiff1d(self._pauli_list, np.arange(4)).size > 0:
            raise ValueError("A pauli index can only be 0, 1, 2, or 3.")
        ls: "npt.NDArray[np.int_]" = np.array([len(x) for x in self._pauli_list])
        if not np.all(ls == ls[0]):
            raise ValueError("All Pauli instructions must have the same length.")
        return cast(int, ls[0])


class GeneralDepolarizingNoise(PauliNoise):
    """Multi qubit general depolarizing noise.

    In the case of multi qubit noise, qubit_indices should specify the qubits on
    which the target gate is acting, in any order, without excess or deficiency.

    Args:
        error_prob: Probability of noise generation.
        qubit_count: Number of qubits of the noise.
        qubit_indices: Sequence of target qubit indices.
        target_gates: Sequence of target gate names.
    """

    def __init__(
        self,
        error_prob: float,
        qubit_count: int,
        qubit_indices: Sequence[int] = (),
        target_gates: Sequence[NonParametricGateNameType] = (),
    ):
        if not qubit_count > 0:
            raise ValueError("qubit_count must be larger than 0.")
        _check_valid_probability(error_prob, "error_prob")
        _check_valid_qubit_indices(qubit_count, qubit_indices)
        term_counts = 4**qubit_count
        prob_identity = 1.0 - error_prob
        prob_pauli = error_prob / (term_counts - 1)
        prob_list = [prob_identity] + (term_counts - 1) * [prob_pauli]
        pauli_list = tuple(it.product([0, 1, 2, 3], repeat=qubit_count))
        super().__init__(
            pauli_list=pauli_list,
            prob_list=prob_list,
            qubit_indices=tuple(qubit_indices),
            target_gates=tuple(target_gates),
        )


class ProbabilisticNoise(GateNoiseInstruction):
    """Multi qubit probabilistic noise.

    This noise is defined by giving matrices representing noise gate operations
    and the probability of each operation explicitly.

    In the case of multi qubit noise, qubit_indices should specify the qubits on
    which the target gate is acting, in any order, without excess or deficiency.

    Args:
        gate_matrices: Sequence of matrices representing gate operations.
        prob_list: Sequence of probability for each operation.
        qubit_indices: Sequence of target qubit indices.
        target_gates: Sequence of target gate names.
        eq_tolerance: Allowed error in the total probability over 1.
    """

    def __init__(
        self,
        gate_matrices: Sequence[Sequence[Sequence[float]]],
        prob_list: Sequence[float],
        qubit_indices: Sequence[int] = (),
        target_gates: Sequence[NonParametricGateNameType] = (),
        eq_tolerance: float = 1.0e-8,
    ):
        if not gate_matrices:
            raise ValueError("gate_matrices cannot be empty.")
        if not prob_list:
            raise ValueError("prob_list cannot be empty.")
        if len(gate_matrices) != len(prob_list):
            raise ValueError("gate_matrices and prob_list must have the same length")
        for i, prob in enumerate(prob_list):
            _check_valid_probability(prob, f"prob_list[{i}]")
        if sum(prob_list) - 1.0 > eq_tolerance:
            raise ValueError("The sum of prob_list must be less than or equal to 1.")
        qubit_count = self._get_qubit_count(gate_matrices)
        self._prob_list, self._gate_matrices = self._get_prob_and_matrix(
            qubit_count, prob_list, gate_matrices
        )
        _check_valid_qubit_indices(qubit_count, qubit_indices)
        super().__init__(
            name="ProbabilisticNoise",
            qubit_count=qubit_count,
            params=(),
            qubit_indices=tuple(qubit_indices),
            target_gates=tuple(target_gates),
        )

    @property
    def prob_list(self) -> Sequence[float]:
        return self._prob_list

    @property
    def gate_matrices(self) -> tuple[npt.NDArray[np.float64], ...]:
        return self._gate_matrices

    def _get_prob_and_matrix(
        self,
        qubit_count: int,
        prob_list: Sequence[float],
        gate_matrices: Sequence[Sequence[Sequence[float]]],
    ) -> tuple[Sequence[float], KrausOperatorSequence]:
        pl = list(prob_list)
        dl: list[npt.NDArray[np.float64]] = [np.array(m) for m in gate_matrices]
        sum_prob = sum(prob_list)

        if sum_prob < 1.0:
            dl.append(np.identity(2**qubit_count))
            pl.append(1.0 - sum_prob)

        return tuple(pl), tuple(dl)

    def _get_qubit_count(
        self, gate_matrices: Sequence[Sequence[Sequence[float]]]
    ) -> int:
        message = (
            "Each gate matrix must be a 2^n by 2^n matrix,"
            " where n is the number of qubits."
        )
        if not _is_aligned_square_matrices(gate_matrices):
            raise ValueError(message)
        qubit_count = np.log2(len(gate_matrices[0]))
        if not qubit_count.is_integer():
            raise ValueError(message)
        return int(qubit_count)


class AbstractKrausNoise(GateNoiseInstruction, ABC):
    """Abstract base class of gate noise instractions which can return their
    Kraus operators."""

    def __init__(
        self,
        name: str,
        qubit_count: int,
        params: tuple[float, ...],
        qubit_indices: tuple[int, ...],
        target_gates: tuple[NonParametricGateNameType, ...],
    ):
        super().__init__(
            name=name,
            qubit_count=qubit_count,
            params=params,
            qubit_indices=qubit_indices,
            target_gates=target_gates,
        )

    @property
    @abstractmethod
    def kraus_operators(self) -> KrausOperatorSequence:
        ...


class KrausNoise(AbstractKrausNoise):
    """Multi qubit Kraus noise.

    In the case of multi qubit noise, qubit_indices should specify the qubits on
    which the target gate is acting, in any order, without excess or deficiency.

    Args:
        kraus_list: Sequence of Kraus operator matrices.
        qubit_indices: Sequence of target qubit indices.
        target_gates: Sequence of target gate names.
    """

    def __init__(
        self,
        kraus_list: Sequence[Sequence[Sequence[float]]],
        qubit_indices: Sequence[int] = (),
        target_gates: Sequence[NonParametricGateNameType] = (),
    ):
        if not kraus_list:
            raise ValueError("kraus_list cannot be empty.")
        qubit_count = self._get_qubit_count(kraus_list)
        self._kraus_operators = self._get_kraus_operator_sequence(kraus_list)
        _check_valid_qubit_indices(qubit_count, qubit_indices)
        super().__init__(
            name="KrausNoise",
            qubit_count=qubit_count,
            params=(),
            qubit_indices=tuple(qubit_indices),
            target_gates=tuple(target_gates),
        )

    @property
    def kraus_operators(self) -> KrausOperatorSequence:
        return self._kraus_operators

    def _get_kraus_operator_sequence(
        self, kraus_list: Sequence[Sequence[Sequence[float]]]
    ) -> KrausOperatorSequence:
        return tuple(np.array(m) for m in kraus_list)

    def _get_qubit_count(self, kraus_list: Sequence[Sequence[Sequence[float]]]) -> int:
        message = (
            "Each Kraus operator must be a 2^n by 2^n matrix,"
            " where n is the number of qubits."
        )
        if not _is_aligned_square_matrices(kraus_list):
            raise ValueError(message)
        qubit_count = np.log2(len(kraus_list[0]))
        if not qubit_count.is_integer():
            raise ValueError(message)
        return int(qubit_count)


class ResetNoise(AbstractKrausNoise):
    r"""Single qubit reset noise.

    Args:
        p0: Reset probability to :math:`|0\rangle`.
        p1: Reset probability to :math:`|1\rangle`.
        qubit_indices: Sequence of target qubit indices.
        target_gates: Sequence of target gate names.
    """

    def __init__(
        self,
        p0: float,
        p1: float,
        qubit_indices: Sequence[int] = (),
        target_gates: Sequence[NonParametricGateNameType] = (),
    ):
        _check_valid_probability(p0, "p0")
        _check_valid_probability(p1, "p1")
        if p0 + p1 > 1:
            raise ValueError("The sum of p0 and p1 must be less than or equal to 1.")
        self._kraus_operators = self._get_kraus_operator_sequence(p0, p1)
        super().__init__(
            name="ResetNoise",
            qubit_count=1,
            params=(p0, p1),
            qubit_indices=tuple(qubit_indices),
            target_gates=tuple(target_gates),
        )

    @property
    def p0(self) -> float:
        return self.params[0]

    @property
    def p1(self) -> float:
        return self.params[1]

    @property
    def kraus_operators(self) -> KrausOperatorSequence:
        return self._kraus_operators

    def _get_kraus_operator_sequence(
        self, p0: float, p1: float
    ) -> KrausOperatorSequence:
        return (
            np.sqrt(1 - p0 - p1) * np.array([[1, 0], [0, 1]]),
            np.sqrt(p0) * np.array([[1, 0], [0, 0]]),
            np.sqrt(p0) * np.array([[0, 1], [0, 0]]),
            np.sqrt(p1) * np.array([[0, 0], [1, 0]]),
            np.sqrt(p1) * np.array([[0, 0], [0, 1]]),
        )


class PhaseDampingNoise(AbstractKrausNoise):
    """Single qubit phase damping noise.

    Args:
        phase_damping_rate: Probability of phase damping.
        qubit_indices: Sequence of target qubit indices.
        target_gates: Sequence of target gate names.
    """

    def __init__(
        self,
        phase_damping_rate: float,
        qubit_indices: Sequence[int] = (),
        target_gates: Sequence[NonParametricGateNameType] = (),
    ):
        _check_valid_probability(phase_damping_rate, "rate")
        self._kraus_operators = self._get_kraus_operator_sequence(phase_damping_rate)
        super().__init__(
            name="PhaseDampingNoise",
            qubit_count=1,
            params=(phase_damping_rate,),
            qubit_indices=tuple(qubit_indices),
            target_gates=tuple(target_gates),
        )

    @property
    def phase_damping_rate(self) -> float:
        return self.params[0]

    @property
    def kraus_operators(self) -> KrausOperatorSequence:
        return self._kraus_operators

    def _get_kraus_operator_sequence(self, rate: float) -> KrausOperatorSequence:
        return (
            np.array([[1, 0], [0, np.sqrt(1 - rate)]]),
            np.array([[0, 0], [0, np.sqrt(rate)]]),
        )


class AmplitudeDampingNoise(AbstractKrausNoise):
    """Single qubit amplitude damping noise.

    Args:
        amplitude_damping_rate: Probability of amplitude damping.
        excited_state_population: Excited state population.
        qubit_indices: Sequence of target qubit indices.
        target_gates: Sequence of target gate names.
    """

    def __init__(
        self,
        amplitude_damping_rate: float,
        excited_state_population: float,
        qubit_indices: Sequence[int] = (),
        target_gates: Sequence[NonParametricGateNameType] = (),
    ):
        _check_valid_probability(amplitude_damping_rate, "rate")
        _check_valid_probability(excited_state_population, "excited_state_population")
        self._kraus_operators = self._get_kraus_operator_sequence(
            amplitude_damping_rate, excited_state_population
        )
        super().__init__(
            name="AmplitudeDampingNoise",
            qubit_count=1,
            params=(amplitude_damping_rate, excited_state_population),
            qubit_indices=tuple(qubit_indices),
            target_gates=tuple(target_gates),
        )

    @property
    def amplitude_damping_rate(self) -> float:
        return self.params[0]

    @property
    def excited_state_population(self) -> float:
        return self.params[1]

    @property
    def kraus_operators(self) -> KrausOperatorSequence:
        return self._kraus_operators

    def _get_kraus_operator_sequence(
        self, rate: float, esp: float
    ) -> KrausOperatorSequence:
        return (
            np.sqrt(1 - esp) * np.array([[1, 0], [0, np.sqrt(1 - rate)]]),
            np.sqrt(1 - esp) * np.array([[0, np.sqrt(rate)], [0, 0]]),
            np.sqrt(esp) * np.array([[np.sqrt(1 - rate), 0], [0, 1]]),
            np.sqrt(esp) * np.array([[0, 0], [np.sqrt(rate), 0]]),
        )


class PhaseAmplitudeDampingNoise(AbstractKrausNoise):
    """Single qubit phase and amplitude damping noise.

    Args:
        phase_damping_rate: Probability of phase damping.
        amplitude_damping_rate: Probability of amplitude damping.
        excited_state_population: Excited state population.
        qubit_indices: Sequence of target qubit indices.
        target_gates: Sequence of target gate names.
    """

    def __init__(
        self,
        phase_damping_rate: float,
        amplitude_damping_rate: float,
        excited_state_population: float,
        qubit_indices: Sequence[int] = (),
        target_gates: Sequence[NonParametricGateNameType] = (),
    ):
        _check_valid_probability(phase_damping_rate, "phase_damping_rate")
        _check_valid_probability(amplitude_damping_rate, "amplitude_damping_rate")
        if phase_damping_rate + amplitude_damping_rate > 1:
            raise ValueError(
                "The sum of phase_damping_rate and amplitude_damping_rate"
                " must be less than or equal to 1."
            )
        _check_valid_probability(excited_state_population, "excited_state_population")
        self._kraus_operators = self._get_kraus_operator_sequence(
            phase_damping_rate, amplitude_damping_rate, excited_state_population
        )
        super().__init__(
            name="PhaseAmplitudeDampingNoise",
            qubit_count=1,
            params=(
                phase_damping_rate,
                amplitude_damping_rate,
                excited_state_population,
            ),
            qubit_indices=tuple(qubit_indices),
            target_gates=tuple(target_gates),
        )

    @property
    def phase_damping_rate(self) -> float:
        return self.params[0]

    @property
    def amplitude_damping_rate(self) -> float:
        return self.params[1]

    @property
    def excited_state_population(self) -> float:
        return self.params[2]

    @property
    def kraus_operators(self) -> KrausOperatorSequence:
        return self._kraus_operators

    def _get_kraus_operator_sequence(
        self, prate: float, arate: float, esp: float
    ) -> KrausOperatorSequence:
        return (
            np.sqrt(1 - esp) * np.array([[1, 0], [0, np.sqrt(1 - arate - prate)]]),
            np.sqrt(1 - esp) * np.array([[0, np.sqrt(arate)], [0, 0]]),
            np.sqrt(1 - esp) * np.array([[0, 0], [0, np.sqrt(prate)]]),
            np.sqrt(esp) * np.array([[np.sqrt(1 - arate - prate), 0], [0, 1]]),
            np.sqrt(esp) * np.array([[0, 0], [np.sqrt(arate), 0]]),
            np.sqrt(esp) * np.array([[np.sqrt(prate), 0], [0, 0]]),
        )


class ThermalRelaxationNoise(AbstractKrausNoise):
    r"""Sigle qubit thermal relaxation noise.

    Args:
        t1: :math:`T_1` relaxation time.
        t2: :math:`T_2` relaxation time.
        gate_time: Gate time.
        excited_state_population: Excited state population.
        qubit_indices: Target qubit indices.
        target_gates: Target gate names.
    """

    def __init__(
        self,
        t1: float,
        t2: float,
        gate_time: float,
        excited_state_population: float,
        qubit_indices: Sequence[int] = (),
        target_gates: Sequence[NonParametricGateNameType] = (),
    ):
        _check_valid_probability(excited_state_population, "excited_state_population")
        if gate_time < 0.0:
            raise ValueError("gate_time must be greater than or equal to 0.")
        if t1 <= 0.0:
            raise ValueError("t1 must be greater than 0.")
        if t2 <= 0.0:
            raise ValueError("t2 must be greater than 0.")
        if t2 > 2.0 * t1:
            raise ValueError("t2 must be less than or equal to 2 * t1.")
        self._kraus_operators = self._get_kraus_operator_sequence(
            t1,
            t2,
            gate_time,
            excited_state_population,
        )
        super().__init__(
            name="ThermalRelaxationNoise",
            qubit_count=1,
            params=(t1, t2, gate_time, excited_state_population),
            qubit_indices=tuple(qubit_indices),
            target_gates=tuple(target_gates),
        )

    @property
    def t1(self) -> float:
        return self.params[0]

    @property
    def t2(self) -> float:
        return self.params[1]

    @property
    def gate_time(self) -> float:
        return self.params[2]

    @property
    def excited_state_population(self) -> float:
        return self.params[3]

    @property
    def kraus_operators(self) -> KrausOperatorSequence:
        return self._kraus_operators

    def _get_kraus_operator_sequence(
        self,
        t1: float,
        t2: float,
        gate_time: float,
        esp: float,
    ) -> KrausOperatorSequence:
        # NOTE
        # Referring to Qiskit.
        # https://qiskit.org/documentation/_modules/qiskit_aer/noise/errors/standard_errors.html#thermal_relaxation_error
        if t1 == np.inf:
            rate1, p_reset = 0.0, 0.0
        else:
            rate1 = 1.0 / t1
            p_reset = 1.0 - np.exp(-gate_time * rate1)
        if t2 == np.inf:
            rate2, exp_t2 = 0.0, 1.0
        else:
            rate2 = 1.0 / t2
            exp_t2 = np.exp(-gate_time * rate2)

        p0, p1 = 1.0 - esp, esp

        choi_matrix: npt.NDArray[np.float64] = np.array(
            [
                [1 - p1 * p_reset, 0, 0, exp_t2],
                [0, p1 * p_reset, 0, 0],
                [0, 0, p0 * p_reset, 0],
                [exp_t2, 0, 0, 1 - p0 * p_reset],
            ]
        )

        eigvals, eigvecs = la.eig(choi_matrix)
        d_matrix = np.sqrt(np.diag(eigvals))
        res = np.dot(np.dot(eigvecs, d_matrix), la.inv(eigvecs))

        return (
            np.transpose(res[:, 0].reshape(2, 2)),
            np.transpose(res[:, 1].reshape(2, 2)),
            np.transpose(res[:, 2].reshape(2, 2)),
            np.transpose(res[:, 3].reshape(2, 2)),
        )
