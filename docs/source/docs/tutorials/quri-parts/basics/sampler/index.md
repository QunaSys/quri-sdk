# Sampler

Unlike statevector simulation, *sampling measurements* are necessary in order to estimate expectation value of operators on a quantum computer. In sampling measurements, execution of a quantum circuit and a subsequent measurement of qubits are repeated multiple times. Estimation of expectation value of operators is then performed using statistics of the repeated measurements.

To perform a sampling measurement of a circuit, you can use a `Sampler`. Here we introduce the definition of `Sampler` and explain how it can be created or executed.

## Prerequisite

QURI Parts modules used in this tutorial: `quri-parts-circuit`, `quri-parts-core` and `quri-parts-qulacs`. You can install them as follows:


```ipython3
!pip install "quri-parts[qulacs]"
```

## Prepare a circuit

As a preparation, we create a circuit to be sampled:


```python
from math import pi
from quri_parts.circuit import QuantumCircuit

# A circuit with 4 qubits
circuit = QuantumCircuit(4)
circuit.add_X_gate(0)
circuit.add_H_gate(1)
circuit.add_Y_gate(2)
circuit.add_CNOT_gate(1, 2)
circuit.add_RX_gate(3, pi/4)
```

## Interface

When performing a sampling measurement for a circuit, you can use a `Sampler`. In QURI Parts, a `Sampler` represents a function that samples a specified (non-parametric) circuit by a specified times and returns the count statistics. In the case of an ideal Sampler, the return value corresponds to probabilities multiplied by shot count.

In the case where sampling from multiple circuits is desired, QURI Parts also provide `ConcurrentSampler`, which is a function that samples from multiple (circuit, shot) pairs.

`Sampler` and `ConcurrentSampler` are both abstract interfaces with the following function signatures:


```python
from typing import Callable, Iterable, Mapping, Union
from typing_extensions import TypeAlias

from quri_parts.circuit import NonParametricQuantumCircuit

#: MeasurementCounts represents count statistics of repeated measurements of a quantum
#: circuit. Keys are observed bit patterns encoded in integers and values are counts
#: of observation of the corresponding bit patterns.
MeasurementCounts: TypeAlias = Mapping[int, Union[int, float]]

#: Sampler represents a function that samples a specified (non-parametric) circuit by
#: a specified times and returns the count statistics. In the case of an ideal Sampler,
# the return value corresponds to probabilities multiplied by shot count.
Sampler: TypeAlias = Callable[[NonParametricQuantumCircuit, int], MeasurementCounts]

#: ConcurrentSampler represents a function that samples specified (non-parametric)
#: circuits concurrently.
ConcurrentSampler: TypeAlias = Callable[
    [Iterable[tuple[NonParametricQuantumCircuit, int]]], Iterable[MeasurementCounts]
]
```

The `Sampler` itself (defined in `quri_parts.core.sampling`) is an abstract interface and you need a concrete instance to actually perform sampling. There are several implementations of `Sampler` interface, some of which use a circuit simulator while others use a real quantum computer.

## Create and execute sampler

Let's create a sampler using state vector simulation with Qulacs and execute sampling with it.


```python
from quri_parts.qulacs.sampler import create_qulacs_vector_sampler

# Create the sampler
sampler = create_qulacs_vector_sampler()
sampling_result = sampler(circuit, shots=1000)
print(sampling_result)
```
```output
Counter({5: 423, 3: 421, 11: 87, 13: 69})
```
`MeasurementCounts` is actually a python `dict` with keys are the observed bit patterns and values are counts.

## List of available samplers

Here is the list of available samplers in QURI Parts. You can set the options for each simulator as the arguments of it's generating function.

| Module | Generating function | Noise | Type |
| --- | --- | :---: | --- |
| quri-parts-itensor | [create_itensor_mps_sampler](https://github.com/QunaSys/quri-parts/blob/b0016537bf24b4a235c181dd468a897605b95b7d/packages/itensor/quri_parts/itensor/sampler.py#L43)([Concurrent](https://github.com/QunaSys/quri-parts/blob/b0016537bf24b4a235c181dd468a897605b95b7d/packages/itensor/quri_parts/itensor/sampler.py#L90)) | ✘ | MPS |
| quri-parts-qulacs | [create_qulacs_vector_sampler](https://github.com/QunaSys/quri-parts/blob/b0016537bf24b4a235c181dd468a897605b95b7d/packages/qulacs/quri_parts/qulacs/sampler.py#L71)([Concurrent](https://github.com/QunaSys/quri-parts/blob/b0016537bf24b4a235c181dd468a897605b95b7d/packages/qulacs/quri_parts/qulacs/sampler.py#L93))<br />[create_qulacs_vector_ideal_sampler](https://github.com/QunaSys/quri-parts/blob/b0016537bf24b4a235c181dd468a897605b95b7d/packages/qulacs/quri_parts/qulacs/sampler.py#L65)<br />[create_qulacs_stochastic_state_vector_sampler](https://github.com/QunaSys/quri-parts/blob/b0016537bf24b4a235c181dd468a897605b95b7d/packages/qulacs/quri_parts/qulacs/sampler.py#L107)([Concurrent](https://github.com/QunaSys/quri-parts/blob/b0016537bf24b4a235c181dd468a897605b95b7d/packages/qulacs/quri_parts/qulacs/sampler.py#L228))<br />[create_qulacs_density_matrix_sampler](https://github.com/QunaSys/quri-parts/blob/b0016537bf24b4a235c181dd468a897605b95b7d/packages/qulacs/quri_parts/qulacs/sampler.py#L128)([Concurrent](https://github.com/QunaSys/quri-parts/blob/b0016537bf24b4a235c181dd468a897605b95b7d/packages/qulacs/quri_parts/qulacs/sampler.py#L212))<br />[create_qulacs_density_matrix_ideal_sampler](https://github.com/QunaSys/quri-parts/blob/b0016537bf24b4a235c181dd468a897605b95b7d/packages/qulacs/quri_parts/qulacs/sampler.py#L152)<br />[create_qulacs_noisesimulator_sampler](https://github.com/QunaSys/quri-parts/blob/b0016537bf24b4a235c181dd468a897605b95b7d/packages/qulacs/quri_parts/qulacs/sampler.py#L171C19-L171C33)([Concurrent](https://github.com/QunaSys/quri-parts/blob/b0016537bf24b4a235c181dd468a897605b95b7d/packages/qulacs/quri_parts/qulacs/sampler.py#L244)) | ✘<br />✘<br />✔<br />✔<br />✔<br />✔ | Statevector<br />Statevector<br />Statevector<br />Density matrix<br />Density matrix<br />[NoiseSimulator](http://docs.qulacs.org/en/latest/api/classNoiseSimulator.html) |
| quri-parts-stim | [create_stim_clifford_sampler](https://github.com/QunaSys/quri-parts/blob/b0016537bf24b4a235c181dd468a897605b95b7d/packages/stim/quri_parts/stim/sampler/__init__.py#L38)([Concurrent](https://github.com/QunaSys/quri-parts/blob/b0016537bf24b4a235c181dd468a897605b95b7d/packages/stim/quri_parts/stim/sampler/__init__.py#L60)) | ✘ | Clifford | 

You can also create a `Sampler` with `create_sampler_from_sampling_backend`([link](https://github.com/QunaSys/quri-parts/blob/b0016537bf24b4a235c181dd468a897605b95b7d/packages/core/quri_parts/core/sampling/__init__.py#L36)) in conjunction with `SamplingBackend`. Below is the list of SamplingBackend available in QURI Parts:

| Module | Backend | Class |
| --- | --- | --- |
| quri-parts-braket | AWS Braket | [BraketSamplingBackend](https://github.com/QunaSys/quri-parts/blob/b0016537bf24b4a235c181dd468a897605b95b7d/packages/braket/quri_parts/braket/backend/sampling.py#L70) |
| quri-parts-qiskit | Qiskit | [QiskitSamplingBackend](https://github.com/QunaSys/quri-parts/blob/b0016537bf24b4a235c181dd468a897605b95b7d/packages/qiskit/quri_parts/qiskit/backend/sampling.py#L69)<br />[QiskitRuntimeSamplingBackend](https://github.com/QunaSys/quri-parts/blob/b0016537bf24b4a235c181dd468a897605b95b7d/packages/qiskit/quri_parts/qiskit/backend/primitive.py#L93) |

For more details please refer to the series of tutorials about sampling using real devices:

- [Sampling Backends](../real_devices/sampling_backends/index.md)
- [Sampling on Braket’s real quantum computers](../real_devices/braket/index.md)
- [Sampling on Qiskit’s real quantum computers](../real_devices/qiskit/index.md)

