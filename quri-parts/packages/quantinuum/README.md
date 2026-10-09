# QURI Parts Quantinuum

QURI Parts Quantinuum is a support library for using Quantinuum with QURI Parts.

It provides

- `quri_parts.quantinuum.circuit`: Quantinuum native gates (`U1q`, `ZZ`, `RZZ`) and transpilers to the Quantinuum gate set.
- `quri_parts.quantinuum.backend`: `QuantinuumSamplingBackend`, a `SamplingBackend` that submits circuits to Quantinuum systems (H-Series, Helios), their emulators and Nexus-hosted simulators through [Quantinuum Nexus](https://docs.quantinuum.com/nexus/).

## Documentation

[QURI Parts Documentation](https://quri-parts.qunasys.com)

## Installation

```
pip install quri-parts-quantinuum
```

To submit jobs through Quantinuum Nexus, install the `nexus` extra:

```
pip install "quri-parts-quantinuum[nexus]"
```

## Usage

```python
import qnexus as qnx
from quri_parts.circuit import QuantumCircuit
from quri_parts.quantinuum.backend import QuantinuumSamplingBackend

qnx.login()  # once per machine: opens a browser, tokens are stored in ~/.qnx/auth

circuit = QuantumCircuit(2)
circuit.add_H_gate(0)
circuit.add_CNOT_gate(0, 1)

backend = QuantinuumSamplingBackend("H2-1E")  # emulator; "H2-1" is the device
job = backend.sample(circuit, shots=1000)
print(job.result().counts)
```

Any Nexus backend config can be passed instead of a device name, e.g. `QuantinuumSamplingBackend(backend_config=qnx.AerConfig())` for a Nexus-hosted simulator or `QuantinuumSamplingBackend(backend_config=qnx.HeliosConfig(system_name="Helios-1E"))` for the Helios emulator.

## License

Apache License 2.0
