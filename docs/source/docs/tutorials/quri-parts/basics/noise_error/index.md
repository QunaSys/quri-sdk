# Noisy Simulation

Quantum circuits running on real machines are affected by a variety of stochastic noises. In QURI Parts, noise models can be defined to represent these noises and reproduce them on simulators. (Qulacs is used in this tutorial.)

## Prerequisite

QURI Parts modules used in this tutorial: `quri-parts-circuit`, `quri-parts-core`, and `quri-parts-qulacs`. You can install them as follows:


```ipython3
!pip install "quri-parts[qulacs]"
```

## Overview

### Prepare a circuit

First, prepare a circuit to apply noise.


```python
from quri_parts.circuit import QuantumCircuit
circuit = QuantumCircuit(3)
circuit.add_H_gate(2)
circuit.add_X_gate(0)
circuit.add_CNOT_gate(2, 1)
circuit.add_Z_gate(2)
```

### Create a noise model

Next, create a noise model. Create several `NoiseInstruction`s that represent noises and their application conditions, and add them to `NoiseModel`.

(This is a noise model to illustrate API functionality and is not a realistic example. Noise models should be adjusted to match the characteristics of the actual equipment of interest.)


```python
import quri_parts.circuit.gate_names as gate_names
from quri_parts.circuit.noise import (
    BitFlipNoise,
    BitPhaseFlipNoise,
    DepolarizingNoise,
    DepthIntervalNoise,
    MeasurementNoise,
    NoiseModel,
    PauliNoise,
    PhaseFlipNoise,
)
noises = [
    # Single qubit noise
    BitFlipNoise(
        error_prob=0.004,
        qubit_indices=[0, 2],  # Qubit 0 or 2
        target_gates=[gate_names.H, gate_names.CNOT],  # H or CNOT gates
    ),
    DepolarizingNoise(
        error_prob=0.003,
        qubit_indices=[],  # All qubits
        target_gates=[gate_names.X, gate_names.CNOT]  # X or CNOT gates
    ),
    PhaseFlipNoise(
        error_prob=0.002,
        qubit_indices=[1, 0],  # Qubit 0 or 1
        target_gates=[]  # All kind of gates
    ),
    BitPhaseFlipNoise(
        error_prob=0.001,
        qubit_indices=[],  # All qubits
        target_gates=[],  # All kind of gates
    ),
    
    # Multi qubit noise
    PauliNoise(
        pauli_list=[[1, 2], [2, 3]],
        prob_list=[0.001, 0.002],
        qubit_indices=[1, 2],  # 2 qubit gates applying to qubits (1, 2) or (2, 1)
        target_gates=[gate_names.CNOT]  # CNOT gates
    ),
    
    # Circuit noise
    DepthIntervalNoise([PhaseFlipNoise(0.001)], depth_interval=5),
    MeasurementNoise([BitFlipNoise(0.004), DepolarizingNoise(0.003)]),
]
model = NoiseModel(noises)
```

For single qubit noises, you can specify the target qubit indices and the target gate names. If the argument is omitted or an empty list is given, all qubits or gates are treated as targets.

The method to specify application condition is similar for multi qubit noises, but the target qubit indices requires the complete set of qubits (in any order) for the target gate. The noise is applied to the qubits sorted as specified in the target gate. For example, if the target gate is specified as CNOT(5, 3), then the noise applied to the qubits (5, 3) not (3, 5).

## Interface

This section is devoted to introduce the `NoiseInstruction` interface and various `NoiseInstruction`s provided by QURI Parts. The `NoiseInstruction` is a type alias that represents 2 types of noise instructions:

- `CircuitNoiseInstruction`: Represents the noise applied depending on the structure of the
    circuit.
- `GateNoiseInstruction`: Represents the noise that is applied when individual gates act on
    qubits.

### `CircuitNoiseInstruction`

A `CircuitNoiseInstruction` represents the noise applied depending on the structure of the circuit. Here are the list of `CircuitNoiseInstruction` QURI Parts provide.

| Name               | Description                                                                                               | Input                                     |
| ------------------ | --------------------------------------------------------------------------------------------------------- | ----------------------------------------- |
| GateIntervalNoise  | For each qubit, given single qubit noises are applied each time a certain number of gates are applied.    | - single_qubit_noises<br />- gate_interval  |
| DepthIntervalNoise | Apply the given single qubit GateNoiseInstruction to all qubits every time a certain depth is advanced.   | - single_qubit_noises<br />- depth_interval |
| MeasurementNoise   | Noise which occurs during the measurement of a qubit| - single_qubit_noises<br />- qubit_indices  |

### `GateNoiseInstruction`

`GateNoiseInstruction` represents the noise that is applied when individual gates act on qubits. It is a dataclass with the following attributes:

- name: Name of the noise.
- qubit_count: Number of qubits this error affects.
- params: Parameters such as error probability, etc. (Depends on the concrete error type.)
- qubit_indices: Indices of qubits this error affects.
- target_gates: Gates affected by this error.

QURI Parts provide several implemetations of `GateNoiseInstruction`s and some special sub-types of them, which we list in later subsections:

#### Basic `GateNoiseInstruction`

Here, we first start with the basic implementations of `GateNoiseInstruction`. They are constructed with 3 parameters:

- error_prob: The probability the noise causes error.
- qubit_indices: The qubit on which the noise can occur. If nothing is passed it, it indicates that the noise can happen on all the qubits in the circuit.
- target_gates: The gates that can generate the noise. If nothing is passed it, it indicates that all the gates are subjected to the noise.

Here we list out these errors:

| Name              | Description                            | Input                                             |
| ----------------- | -------------------------------------- | ------------------------------------------------- |
| BitFlipNoise      | Single qubit bit flip noise            | - error_prob<br />- qubit_indices<br />- target_gates |
| PhaseFlipNoise    | Single qubit phase flip noise          | - error_prob<br />- qubit_indices<br />- target_gates |
| BitPhaseFlipNoise | Single qubit bit and phase flip noise. | - error_prob<br />- qubit_indices<br />- target_gates |
| DepolarizingNoise | Single qubit depolarizing noise.       | - error_prob<br />- qubit_indices<br />- target_gates |

#### `PauliNoise`

`PauliNoise` is a subtype of `GateNoiseInstruction` that involves Pauli gates acting on multiple qubits. We summarize the `PauliNoise` that QURI Parts provide. Note that in the input column, we omit the arguments `qubit_indices` and `target_gates` as all the noise instructions require them. We also include the formula of the density matrix after the noise is applied to the state.

| Name                     | Description                            | Input                                             |   Density matrix after noise|
| ------------------------ | -------------------------------------- | ------------------------------------------------- | --- |
| PauliNoise               | Multi qubit Pauli noise.               | - pauli_list<br />- prob_list<br />- eq_tolerance |  $ \sum_{i}p_i P_{i} \rho P_{i} + (1-\sum_{i}p_i)\rho $ |
| GeneralDepolarizingNoise | Multi qubit general depolarizing noise | - error_prob<br />- qubit_count                   |  $\frac{p}{4^n - 1} \sum_{i_1=0}^3 \cdots \sum_{i_n=0}^3 E_{i_1 \cdots i_n} \rho E_{i_1 \cdots i_n} + (1-p)\rho$ |


Here note that in the `GeneralDepolarizingNoise`, the operator $E_{i_1\cdots i_n}$ is given by products of Pauli matrices:

$$
\begin{equation}
    E_{i_i\cdots i_n} = X_{i_1} \cdots X_{i_n}
\end{equation}
$$
And note that the summation in the first term does not include $\{i_1, \cdots, i_n \} = \{0, \cdots, 0\}$.

#### Kraus Noises

We also provide another sub-type of `GateNoiseInstruction`: the `AbstractKrausNoise`. It is a `GateNoiseInstruction` with a `kraus_operators` property which returns the list of explicit [Kraus operator](https://en.wikipedia.org/wiki/Quantum_operation#Kraus_operators) matrices that defines the noise. Here, we list out all of them. Note that in the "input" column, we omit the `qubit_indices` and `target_gates` arguments as all the `GateNoiseInstruction` require them.


| Name                       | Description                                     | Input                                                                                              |
| -------------------------- | ----------------------------------------------- | -------------------------------------------------------------------------------------------------- |
| KrausNoise                 | Multi qubit Kraus noise                         | - kraus_list: list of Kraus operator matrices                                                      |
| ResetNoise                 | Single qubit reset noise.                       | - p0: Probability of resetting to $\|0\rangle$<br />- p1: Probability of resetting to $\|1\rangle$ |
| PhaseDampingNoise          | Single qubit phase damping noise.               | - phase_damping_rate                                                                               |
| AmplitudeDampingNoise      | Single qubit amplitude damping noise.           | - amplitude_damping_rate<br />- excited_state_population                                             |
| PhaseAmplitudeDampingNoise | Single qubit phase and amplitude damping noise. | - phase_damping_rate<br />- amplitude_damping_rate<br />- excited_state_population                     |
| ThermalRelaxationNoise     | Sigle qubit thermal relaxation noise.           | - t1<br />- t2<br />- gate_time<br />- excited_state_population                                          |

For example, let's look at the reset noise:


```python
from quri_parts.circuit.noise import ResetNoise

reset_noise = ResetNoise(p0=0.04, p1=0.16)
reset_noise.kraus_operators
```



```output
[[[0.8944271909999159, 0.0], [0.0, 0.8944271909999159]],
 [[0.2, 0.0], [0.0, 0.0]],
 [[0.0, 0.2], [0.0, 0.0]],
 [[0.0, 0.0], [0.4, 0.0]],
 [[0.0, 0.0], [0.0, 0.4]]]
```


### `NoiseModel`

Finally, we introduce the `NoiseModel` object. It is the object in QURI Parts that represents a noise model containing multiple `GateNoiseInstruction`s and `CircuitNoiseInstrction`s. It is also the object that is passed around to create noisy circuit in QURI Parts. A `NoiseModel` is created by a sequence of `NoiseInstruction`s. Please note that the noise do not commute with each other, so that the order in the sequence of `NoiseInstruction` matters.

It also provides some convenient methods for us to modify our model. For example, we can use `add_noise` and `extend` to add noise instructions to the model:


```python
model = NoiseModel([
    BitFlipNoise(
        error_prob=0.004,
        qubit_indices=[0, 2],
        target_gates=[gate_names.H, gate_names.CNOT],
    ),
    DepolarizingNoise(
        error_prob=0.003,
        target_gates=[gate_names.X, gate_names.CNOT]
    ),
])

# Add a single instruction
model.add_noise(PhaseFlipNoise(error_prob=0.002, qubit_indices=[1, 0]))
model.add_noise(BitPhaseFlipNoise(error_prob=0.001,))
    
# Add a sequence of instructions
model.extend([
    PauliNoise(
        pauli_list=[[1, 2], [2, 3]],
        prob_list=[0.001, 0.002],
        qubit_indices=[1, 2],
        target_gates=[gate_names.CNOT]
    ),
    DepthIntervalNoise([PhaseFlipNoise(0.001)], depth_interval=5),
    MeasurementNoise([BitFlipNoise(0.004), DepolarizingNoise(0.003)])
])
```

We also provide `.noises_for_circuit` to inspect the `CircuitNoiseInstrcution`s in the model. `.noises_for_gate` identify the gate noise applied to a gate on a particular


```python
from quri_parts.circuit import X

print("Circuit noises:", model.noises_for_circuit())

print("")
print("Gate noise on X(0):")
for noise in model.noises_for_gate(X(0)):
    print(noise)
```
```output
Circuit noises: <quri_parts.rust.circuit.noise.CircuitNoiseInstance object at 0x7f8973991760>

Gate noise on X(0):
((0,), <quri_parts.rust.circuit.noise.GateNoiseInstruction object at 0x7f897396f130>)
((0,), <quri_parts.rust.circuit.noise.GateNoiseInstruction object at 0x7f897396e4f0>)
((0,), <quri_parts.rust.circuit.noise.GateNoiseInstruction object at 0x7f897396f210>)
```
## Simulating noisy system with Qulacs

### Convert circuit with noise model

If you need a Qulacs circuit with the noise model applied directly, the following circuit conversion function is provided. For purposes such as sampling or estimation, it is usually not necessary for the user to perform this conversion. However, if you choose to work with Qulacs directly, please check out the [Qulacs tutorial for noisy simulation](https://docs.qulacs.org/en/latest/guide/2.0_python_advanced.html#NoiseSimulator).


```python
from quri_parts.qulacs.circuit.noise import convert_circuit_with_noise_model
qulacs_circuit = convert_circuit_with_noise_model(circuit, model)
```

### Sampling simulation with Qulacs

For sampling, several functions are provided to create a Sampler with a noise model applied.


```python
from quri_parts.qulacs.sampler import create_qulacs_density_matrix_sampler
density_matrix_sampler = create_qulacs_density_matrix_sampler(model)
counts = density_matrix_sampler(circuit, shots=1000)
counts
```



```output
Counter({1: 488, 7: 482, 3: 13, 5: 11, 0: 4, 6: 2})
```



```python
from quri_parts.qulacs.sampler import create_qulacs_stochastic_state_vector_sampler
stochastic_state_vector_sampler = create_qulacs_stochastic_state_vector_sampler(model)
counts = stochastic_state_vector_sampler(circuit, shots=1000)
counts
```



```output
Counter({7: 493, 1: 475, 3: 14, 5: 10, 6: 6, 0: 2})
```


Density matrix sampler (created by `create_qulacs_density_matrix_sampler()`) uses the density matrix for calculating measurement probability for sampling, while stochastic state vector sampler (created by `create_qulacs_stochastic_state_vector_sampler()`) performs sampling by repeating stochastic state vector simulation for a specified shot count. The computation time varies with the type of circuit, but in general, stochastic state vector sampler is advantageous when the number of shots is less than about 10^3.

The usage of `Sampler` with noise model is the same as that of other `Sampler`, except that a `NoiseModel` should be given on creation. As with regular `Sampler`, there are versions that support concurrent execution. Please refer to the API documentation and [the sampler tutorial](../sampler/index.md) for details on how to use `Sampler`.

### Estimation of operator expectation value with Qulacs

Also for estimation, you can create an `Estimator` with noise model applied.


```python
from quri_parts.qulacs.estimator import create_qulacs_density_matrix_estimator
density_matrix_estimator = create_qulacs_density_matrix_estimator(model)
```

Simillar to the `Sampler` case, the usage of `Estimator` with noise model is the same as that of other `Estimator`s, except that a `NoiseModel` should be given on creation. As with regular `Estimator`, there are versions that support parametric circuit and/or concurrent execution. Please refer to the API documentation and [Estimator tutorial](../estimators/index.md) for details on how to use `Estimator`.

Finally, let's take a simple example to see the noise model is working. Create a circuit with only one X gate applied and calculate the expectation value with an empty noise model.


```python
from quri_parts.core.operator import pauli_label
from quri_parts.core.state import quantum_state

circuit = QuantumCircuit(1)
circuit.add_X_gate(0)
state = quantum_state(1, circuit=circuit)

pauli = pauli_label("Z0")

empty_model = NoiseModel()

estimator = create_qulacs_density_matrix_estimator(empty_model)
estimate = estimator(pauli, state)
estimate.value
```



```output
(-1+0j)
```


The result is as expected. Now let's add a bit flip noise with probability 0.5 to the noise model.


```python
bitflip_model = NoiseModel([BitFlipNoise(0.5)])

noised_estimator = create_qulacs_density_matrix_estimator(bitflip_model)
noised_estimate = noised_estimator(pauli, state)
noised_estimate.value
```



```output
0j
```


We are getting the expected effect.
