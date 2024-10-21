import qulacs

from quri_parts.circuit import ImmutableQuantumCircuit
from quri_parts.rust.circuit.noise import NoiseModel

def convert_circuit(gate: ImmutableQuantumCircuit) -> qulacs.QuantumCircuit: ...
def convert_circuit_with_noise_model(
    circuit: ImmutableQuantumCircuit, noise_model: NoiseModel
) -> qulacs.QuantumCircuit: ...
