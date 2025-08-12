# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from .noise_instruction import (
    AmplitudeDampingNoise,
    BitFlipNoise,
    BitPhaseFlipNoise,
    CircuitNoiseInstruction,
    DepolarizingNoise,
    DepthIntervalNoise,
    GateIntervalNoise,
    GateNoiseInstruction,
    GeneralDepolarizingNoise,
    KrausNoise,
    MeasurementNoise,
    NoiseInstruction,
    PauliNoise,
    PhaseAmplitudeDampingNoise,
    PhaseDampingNoise,
    PhaseFlipNoise,
    ProbabilisticNoise,
    QubitNoisePair,
    ResetNoise,
    ThermalRelaxationNoise,
)
from .noise_model import CircuitNoiseInstance, NoiseModel

#: Represents a backend-independent noise instruction to be added to NoiseModel.
NoiseInstruction = NoiseInstruction
QubitNoisePair = QubitNoisePair

__all__ = [
    "AmplitudeDampingNoise",
    "BitFlipNoise",
    "BitPhaseFlipNoise",
    "CircuitNoiseInstance",
    "CircuitNoiseInstruction",
    "DepolarizingNoise",
    "DepthIntervalNoise",
    "GateIntervalNoise",
    "GateNoiseInstruction",
    "GeneralDepolarizingNoise",
    "KrausNoise",
    "MeasurementNoise",
    "NoiseModel",
    "NoiseInstruction",
    "PauliNoise",
    "PhaseAmplitudeDampingNoise",
    "PhaseDampingNoise",
    "PhaseFlipNoise",
    "ProbabilisticNoise",
    "QubitNoisePair",
    "ResetNoise",
    "ThermalRelaxationNoise",
]
