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

# from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Union, cast

import numpy as np
import numpy.linalg as la
import numpy.typing as npt
from typing_extensions import TypeAlias

from quri_parts.circuit.gate_names import NonParametricGateNameType
from quri_parts.rust.circuit.noise import (
    DepthIntervalNoise,
    GateIntervalNoise,
    GateNoiseInstruction,
    MeasurementNoise,
)

#: Represents a backend-independent noise instruction to be added to NoiseModel.
NoiseInstruction: TypeAlias = Union[
    GateNoiseInstruction, GateIntervalNoise, DepthIntervalNoise, MeasurementNoise
]
CircuitNoiseInstruction: TypeAlias = Union[
    GateIntervalNoise, DepthIntervalNoise, MeasurementNoise
]
KrausOperatorSequence: TypeAlias = tuple[Sequence[Sequence[float]], ...]
GateMatrices: TypeAlias = tuple[Sequence[Sequence[float]], ...]

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


def BitFlipNoise(
    error_prob: float,
    qubit_indices: Sequence[int] = (),
    target_gates: Sequence[NonParametricGateNameType] = (),
) -> GateNoiseInstruction:
    """Single qubit bit flip noise.

    Args:
        error_prob: Probability of noise generation.
        qubit_indices: Sequence of target qubit indices.
        target_gates: Sequence of target gate names.
    """

    return GateNoiseInstruction(
        name="BitFlipNoise",
        qubit_count=1,
        params=(error_prob,),
        qubit_indices=tuple(qubit_indices),
        target_gates=tuple(target_gates),
    )


def PhaseFlipNoise(
    error_prob: float,
    qubit_indices: Sequence[int] = (),
    target_gates: Sequence[NonParametricGateNameType] = (),
) -> GateNoiseInstruction:
    """Single qubit phase flip noise.

    Args:
        error_prob: Probability of noise generation.
        qubit_indices: Sequence of target qubit indices.
        target_gates: Sequence of target gate names.
    """
    return GateNoiseInstruction(
        name="PhaseFlipNoise",
        qubit_count=1,
        params=(error_prob,),
        qubit_indices=tuple(qubit_indices),
        target_gates=tuple(target_gates),
    )


def BitPhaseFlipNoise(
    error_prob: float,
    qubit_indices: Sequence[int] = (),
    target_gates: Sequence[NonParametricGateNameType] = (),
) -> GateNoiseInstruction:
    """Single qubit bit and phase flip noise.

    Args:
        error_prob: Probability of noise generation.
        qubit_indices: Sequence of target qubit indices.
        target_gates: Sequence of target gate names.
    """
    return GateNoiseInstruction(
        name="BitPhaseFlipNoise",
        qubit_count=1,
        params=(error_prob,),
        qubit_indices=tuple(qubit_indices),
        target_gates=tuple(target_gates),
    )


def DepolarizingNoise(
    error_prob: float,
    qubit_indices: Sequence[int] = (),
    target_gates: Sequence[NonParametricGateNameType] = (),
) -> GateNoiseInstruction:
    """Single qubit depolarizing noise.

    Args:
        error_prob: Probability of noise generation.
        qubit_indices: Sequence of target qubit indices.
        target_gates: Sequence of target gate names.
    """
    return GateNoiseInstruction(
        name="DepolarizingNoise",
        qubit_count=1,
        params=(error_prob,),
        qubit_indices=tuple(qubit_indices),
        target_gates=tuple(target_gates),
    )


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


def PauliNoise(
    pauli_list: Sequence[Sequence[int]],
    prob_list: Sequence[float],
    qubit_indices: Sequence[int] = (),
    target_gates: Sequence[NonParametricGateNameType] = (),
    eq_tolerance: float = 1.0e-8,
    name: str = "PauliNoise",
) -> GateNoiseInstruction:
    """Multi qubit Pauli noise.

    In the case of multi qubit noise, qubit_indices should specify the qubits on
    which the target gate is acting, in any order, without excess or deficiency.

    Args:
        pauli_list: Sequence of series of Pauli ids.
        prob_list: Sequence of probability for each Pauli operations.
        qubit_indices: Sequence of target qubit indices.
        target_gates: Sequence of target gate names.
        eq_tolerance: Allowed error in the total probability over 1.
        name: Give Specified name for the noise.
    """

    def _get_qubit_count(pauli_list: Sequence[Sequence[int]]) -> int:
        if np.setdiff1d(pauli_list, np.arange(4)).size > 0:
            raise ValueError("A pauli index can only be 0, 1, 2, or 3.")
        ls: "npt.NDArray[np.int_]" = np.array([len(x) for x in pauli_list])
        if not np.all(ls == ls[0]):
            raise ValueError("All Pauli instructions must have the same length.")
        return cast(int, ls[0])

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

    qubit_count = _get_qubit_count(tuple(pauli_list))
    _check_valid_qubit_indices(qubit_count, qubit_indices)

    return GateNoiseInstruction(
        name=name,
        qubit_count=qubit_count,
        params=(),
        qubit_indices=tuple(qubit_indices),
        target_gates=tuple(target_gates),
        pauli_list=tuple(pauli_list),
        prob_list=tuple(prob_list),
    )


def GeneralDepolarizingNoise(
    error_prob: float,
    qubit_count: int,
    qubit_indices: Sequence[int] = (),
    target_gates: Sequence[NonParametricGateNameType] = (),
) -> GateNoiseInstruction:
    """Multi qubit general depolarizing noise.

    In the case of multi qubit noise, qubit_indices should specify the qubits on
    which the target gate is acting, in any order, without excess or deficiency.

    Args:
        error_prob: Probability of noise generation.
        qubit_count: Number of qubits of the noise.
        qubit_indices: Sequence of target qubit indices.
        target_gates: Sequence of target gate names.
    """
    if not qubit_count > 0:
        raise ValueError("qubit_count must be larger than 0.")
    _check_valid_probability(error_prob, "error_prob")
    _check_valid_qubit_indices(qubit_count, qubit_indices)

    term_counts = 4**qubit_count
    prob_identity = 1.0 - error_prob
    prob_pauli = error_prob / (term_counts - 1)
    prob_list = [prob_identity] + (term_counts - 1) * [prob_pauli]
    pauli_list = tuple(it.product([0, 1, 2, 3], repeat=qubit_count))

    return PauliNoise(
        pauli_list=pauli_list,
        prob_list=prob_list,
        qubit_indices=tuple(qubit_indices),
        target_gates=tuple(target_gates),
        name="GeneralDepolarizingNoise",
    )


def ProbabilisticNoise(
    gate_matrices: Sequence[Sequence[Sequence[float]]],
    prob_list: Sequence[float],
    qubit_indices: Sequence[int] = (),
    target_gates: Sequence[NonParametricGateNameType] = (),
    eq_tolerance: float = 1.0e-8,
) -> GateNoiseInstruction:
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

    def _get_prob_and_matrix(
        qubit_count: int,
        prob_list: Sequence[float],
        gate_matrices: Sequence[Sequence[Sequence[float]]],
    ) -> tuple[Sequence[float], KrausOperatorSequence]:
        pl = list(prob_list)
        # dl: list[npt.NDArray[np.float64]] = [np.array(m) for m in gate_matrices]
        dl = list(gate_matrices)
        sum_prob = sum(prob_list)

        if sum_prob < 1.0:
            dl.append(np.identity(2**qubit_count).tolist())
            pl.append(1.0 - sum_prob)

        return tuple(pl), tuple(dl)

    def _get_qubit_count(gate_matrices: Sequence[Sequence[Sequence[float]]]) -> int:
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

    qubit_count = _get_qubit_count(gate_matrices)
    _prob_list, _gate_matrices = _get_prob_and_matrix(
        qubit_count, prob_list, gate_matrices
    )
    _check_valid_qubit_indices(qubit_count, qubit_indices)

    return GateNoiseInstruction(
        name="ProbabilisticNoise",
        qubit_count=qubit_count,
        params=(),
        qubit_indices=tuple(qubit_indices),
        target_gates=tuple(target_gates),
        prob_list=_prob_list,
        gate_matrices=_gate_matrices,
    )


def KrausNoise(
    kraus_list: Sequence[Sequence[Sequence[float]]],
    qubit_indices: Sequence[int] = (),
    target_gates: Sequence[NonParametricGateNameType] = (),
) -> GateNoiseInstruction:
    """Multi qubit Kraus noise.

    In the case of multi qubit noise, qubit_indices should specify the qubits on
    which the target gate is acting, in any order, without excess or deficiency.

    Args:
        kraus_list: Sequence of Kraus operator matrices.
        qubit_indices: Sequence of target qubit indices.
        target_gates: Sequence of target gate names.
    """

    def _get_qubit_count(kraus_list: Sequence[Sequence[Sequence[float]]]) -> int:
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

    if not kraus_list:
        raise ValueError("kraus_list cannot be empty.")

    qubit_count = _get_qubit_count(kraus_list)
    _check_valid_qubit_indices(qubit_count, qubit_indices)

    return GateNoiseInstruction(
        name="KrausNoise",
        qubit_count=qubit_count,
        params=(),
        qubit_indices=tuple(qubit_indices),
        target_gates=tuple(target_gates),
        kraus_operators=kraus_list,
    )


def ResetNoise(
    p0: float,
    p1: float,
    qubit_indices: Sequence[int] = (),
    target_gates: Sequence[NonParametricGateNameType] = (),
) -> GateNoiseInstruction:
    r"""Single qubit reset noise.

    Args:
        p0: Reset probability to :math:`|0\rangle`.
        p1: Reset probability to :math:`|1\rangle`.
        qubit_indices: Sequence of target qubit indices.
        target_gates: Sequence of target gate names.
    """

    def _get_kraus_operator_sequence(p0: float, p1: float) -> KrausOperatorSequence:
        return (
            np.sqrt(1 - p0 - p1) * np.array([[1, 0], [0, 1]]),
            np.sqrt(p0) * np.array([[1, 0], [0, 0]]),
            np.sqrt(p0) * np.array([[0, 1], [0, 0]]),
            np.sqrt(p1) * np.array([[0, 0], [1, 0]]),
            np.sqrt(p1) * np.array([[0, 0], [0, 1]]),
        )

    _check_valid_probability(p0, "p0")
    _check_valid_probability(p1, "p1")
    if p0 + p1 > 1:
        raise ValueError("The sum of p0 and p1 must be less than or equal to 1.")
    _kraus_operators = _get_kraus_operator_sequence(p0, p1)

    return GateNoiseInstruction(
        name="ResetNoise",
        qubit_count=1,
        params=(p0, p1),
        qubit_indices=tuple(qubit_indices),
        target_gates=tuple(target_gates),
        kraus_operators=_kraus_operators,
    )


def PhaseDampingNoise(
    phase_damping_rate: float,
    qubit_indices: Sequence[int] = (),
    target_gates: Sequence[NonParametricGateNameType] = (),
) -> GateNoiseInstruction:
    """Single qubit phase damping noise.

    Args:
        phase_damping_rate: Probability of phase damping.
        qubit_indices: Sequence of target qubit indices.
        target_gates: Sequence of target gate names.
    """

    def _get_kraus_operator_sequence(rate: float) -> KrausOperatorSequence:
        return (
            [[1, 0], [0, np.sqrt(1 - rate)]],
            [[0, 0], [0, np.sqrt(rate)]],
        )

    _check_valid_probability(phase_damping_rate, "rate")
    _kraus_operators = _get_kraus_operator_sequence(phase_damping_rate)

    return GateNoiseInstruction(
        name="PhaseDampingNoise",
        qubit_count=1,
        params=(phase_damping_rate,),
        qubit_indices=tuple(qubit_indices),
        target_gates=tuple(target_gates),
        kraus_operators=_kraus_operators,
    )


def AmplitudeDampingNoise(
    amplitude_damping_rate: float,
    excited_state_population: float,
    qubit_indices: Sequence[int] = (),
    target_gates: Sequence[NonParametricGateNameType] = (),
) -> GateNoiseInstruction:
    """Single qubit amplitude damping noise.

    Args:
        amplitude_damping_rate: Probability of amplitude damping.
        excited_state_population: Excited state population.
        qubit_indices: Sequence of target qubit indices.
        target_gates: Sequence of target gate names.
    """

    def _get_kraus_operator_sequence(rate: float, esp: float) -> KrausOperatorSequence:
        return (
            np.sqrt(1 - esp) * np.array([[1, 0], [0, np.sqrt(1 - rate)]]),
            np.sqrt(1 - esp) * np.array([[0, np.sqrt(rate)], [0, 0]]),
            np.sqrt(esp) * np.array([[np.sqrt(1 - rate), 0], [0, 1]]),
            np.sqrt(esp) * np.array([[0, 0], [np.sqrt(rate), 0]]),
        )

    _check_valid_probability(amplitude_damping_rate, "rate")
    _check_valid_probability(excited_state_population, "excited_state_population")
    _kraus_operators = _get_kraus_operator_sequence(
        amplitude_damping_rate, excited_state_population
    )

    return GateNoiseInstruction(
        name="AmplitudeDampingNoise",
        qubit_count=1,
        params=(amplitude_damping_rate, excited_state_population),
        qubit_indices=tuple(qubit_indices),
        target_gates=tuple(target_gates),
        kraus_operators=_kraus_operators,
    )


def PhaseAmplitudeDampingNoise(
    phase_damping_rate: float,
    amplitude_damping_rate: float,
    excited_state_population: float,
    qubit_indices: Sequence[int] = (),
    target_gates: Sequence[NonParametricGateNameType] = (),
) -> GateNoiseInstruction:
    """Single qubit phase and amplitude damping noise.

    Args:
        phase_damping_rate: Probability of phase damping.
        amplitude_damping_rate: Probability of amplitude damping.
        excited_state_population: Excited state population.
        qubit_indices: Sequence of target qubit indices.
        target_gates: Sequence of target gate names.
    """

    def _get_kraus_operator_sequence(
        prate: float, arate: float, esp: float
    ) -> KrausOperatorSequence:
        return (
            np.sqrt(1 - esp) * np.array([[1, 0], [0, np.sqrt(1 - arate - prate)]]),
            np.sqrt(1 - esp) * np.array([[0, np.sqrt(arate)], [0, 0]]),
            np.sqrt(1 - esp) * np.array([[0, 0], [0, np.sqrt(prate)]]),
            np.sqrt(esp) * np.array([[np.sqrt(1 - arate - prate), 0], [0, 1]]),
            np.sqrt(esp) * np.array([[0, 0], [np.sqrt(arate), 0]]),
            np.sqrt(esp) * np.array([[np.sqrt(prate), 0], [0, 0]]),
        )

    _check_valid_probability(phase_damping_rate, "phase_damping_rate")
    _check_valid_probability(amplitude_damping_rate, "amplitude_damping_rate")
    if phase_damping_rate + amplitude_damping_rate > 1:
        raise ValueError(
            "The sum of phase_damping_rate and amplitude_damping_rate"
            " must be less than or equal to 1."
        )

    _check_valid_probability(excited_state_population, "excited_state_population")
    _kraus_operators = _get_kraus_operator_sequence(
        phase_damping_rate, amplitude_damping_rate, excited_state_population
    )

    return GateNoiseInstruction(
        name="PhaseAmplitudeDampingNoise",
        qubit_count=1,
        params=(
            phase_damping_rate,
            amplitude_damping_rate,
            excited_state_population,
        ),
        qubit_indices=tuple(qubit_indices),
        target_gates=tuple(target_gates),
        kraus_operators=_kraus_operators,
    )


def ThermalRelaxationNoise(
    t1: float,
    t2: float,
    gate_time: float,
    excited_state_population: float,
    qubit_indices: Sequence[int] = (),
    target_gates: Sequence[NonParametricGateNameType] = (),
) -> GateNoiseInstruction:
    r"""Sigle qubit thermal relaxation noise.

    Args:
        t1: :math:`T_1` relaxation time.
        t2: :math:`T_2` relaxation time.
        gate_time: Gate time.
        excited_state_population: Excited state population.
        qubit_indices: Target qubit indices.
        target_gates: Target gate names.
    """

    def _get_kraus_operator_sequence(
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
            np.transpose(res[:, 0].reshape(2, 2)).tolist(),
            np.transpose(res[:, 1].reshape(2, 2)).tolist(),
            np.transpose(res[:, 2].reshape(2, 2)).tolist(),
            np.transpose(res[:, 3].reshape(2, 2)).tolist(),
        )

    _check_valid_probability(excited_state_population, "excited_state_population")
    if gate_time < 0.0:
        raise ValueError("gate_time must be greater than or equal to 0.")
    if t1 <= 0.0:
        raise ValueError("t1 must be greater than 0.")
    if t2 <= 0.0:
        raise ValueError("t2 must be greater than 0.")
    if t2 > 2.0 * t1:
        raise ValueError("t2 must be less than or equal to 2 * t1.")

    _kraus_operators = _get_kraus_operator_sequence(
        t1,
        t2,
        gate_time,
        excited_state_population,
    )

    return GateNoiseInstruction(
        name="ThermalRelaxationNoise",
        qubit_count=1,
        params=(t1, t2, gate_time, excited_state_population),
        qubit_indices=tuple(qubit_indices),
        target_gates=tuple(target_gates),
        kraus_operators=_kraus_operators,
    )


__all__ = [
    "AmplitudeDampingNoise",
    "BitFlipNoise",
    "BitPhaseFlipNoise",
    "CircuitNoiseInstruction",
    "DepolarizingNoise",
    "DepthIntervalNoise",
    "GateIntervalNoise",
    "GateNoiseInstruction",
    "GeneralDepolarizingNoise",
    "KrausNoise",
    "MeasurementNoise",
    "NoiseInstruction",
    "PhaseFlipNoise",
    "PauliNoise",
    "PhaseAmplitudeDampingNoise",
    "PhaseDampingNoise",
    "ProbabilisticNoise",
    "ResetNoise",
    "ThermalRelaxationNoise",
]
