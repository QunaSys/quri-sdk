# Readout Error Mitigation

Due to the imperfection of real devices, errors occur in state preparation and measurement. Readout error mitigation reduces the effect of those errors by applying inverse of such noisy operations. The inverse of the noisy operation, here we call filter matrix, plays an important role in readout error mitigation. Noisy counts $C_{\text{noisy}}$ can be considered as a product of ideal counts $C_{\text{ideal}}$ which we could get in noiseless world and the error matrix $E$.

$$
C_{\text{noisy}} = EC_{\text{ideal}}
$$

The $(i, j)$ components of $E$ represent the probability that the true state is $|j\rangle$ when the measurement reports $|i\rangle$ as the outcome, which is represented as $P(i|j)$. That is, summing vertically over the matrix $E$ yields 1. 
For exsample, $E$ of 2 qubits device is

$$
E = \left(
\begin{matrix} 
P(00|00) & P(00|01) & P(00|10) & P(00|11) \\ 
P(01|00) & P(01|01) & P(01|10) & P(01|11) \\ 
P(10|00) & P(10|01) & P(10|10) & P(10|11) \\ 
P(11|00) & P(11|01) & P(11|10) & P(11|11) 
\end{matrix} 
\right)
$$

$E$ is estimated by repeating state preparation and measurement in each basis and obtaining probability distributions from histograms of actual measurement results.

Filter matrix is defined as the inverse of error matrix. With filter matrix we can estimate the error-free counts

$$
C_{\text{ideal}} = E^{-1}C_{\text{noisy}}.
$$

In this tutorial, we demonstrate how to build the filter matrix and predict the noise-free sampling count.

## Prerequisite

QURI Parts modules used in this tutorial: `quri-parts-algo`, `quri-parts-circuit`, `quri-parts-core` and `quri-parts-qulacs`. You can install them as follows:


```python
# !pip install "quri-parts[qulacs]"
```

## Preparation and overview

Here, we prepare the circuit and the noise model we use throughout this tutorial. The circuit we use in this tutorial consists of an identity part and a non-trivial part. The non-trivial part is responsible for converting the state $|000\rangle$ into  $\frac{1}{\sqrt{2}}\left(|000\rangle + |111\rangle\right)$, while we decompose the identity circuit into multiple gates to amplify the effect of the noises.


```python
from quri_parts.circuit import QuantumCircuit
from quri_parts.circuit.utils.circuit_drawer import draw_circuit

qubit_count = 3

identity_circuit = QuantumCircuit(3)
identity_circuit.add_RX_gate(0, 1.3)
identity_circuit.add_RY_gate(1, 0.2)
identity_circuit.add_RZ_gate(0, -2.3)
identity_circuit.add_SqrtXdag_gate(1)
identity_circuit.add_T_gate(0)
identity_circuit.add_RX_gate(1, 0.4)
identity_circuit.add_RY_gate(0, 2.7)
identity_circuit.add_Tdag_gate(1)
identity_circuit.add_RY_gate(0, -2.7)
identity_circuit.add_T_gate(1)
identity_circuit.add_Tdag_gate(0)
identity_circuit.add_RX_gate(1, -0.4)
identity_circuit.add_RZ_gate(0, 2.3)
identity_circuit.add_SqrtX_gate(1)
identity_circuit.add_RX_gate(0, -1.3)
identity_circuit.add_RY_gate(1, -0.2)

circuit = QuantumCircuit(3)
circuit += identity_circuit
circuit.add_H_gate(0)
circuit.add_CNOT_gate(0, 1)
circuit.add_CNOT_gate(0, 2)


print("The circuit:")
draw_circuit(circuit, line_length=200)
```
```output
The circuit:
   ___     ___     ___     ___     ___     ___     ___     ___     ___                  
  |RX |   |RZ |   | T |   |RY |   |RY |   |Tdg|   |RZ |   |RX |   | H |                 
--|0  |---|2  |---|4  |---|6  |---|8  |---|10 |---|12 |---|14 |---|16 |-----●-------●---
  |___|   |___|   |___|   |___|   |___|   |___|   |___|   |___|   |___|     |       |   
   ___     ___     ___     ___     ___     ___     ___     ___             _|_      |   
  |RY |   |sXd|   |RX |   |Tdg|   | T |   |RX |   |sqX|   |RY |           |CX |     |   
--|1  |---|3  |---|5  |---|7  |---|9  |---|11 |---|13 |---|15 |-----------|17 |-----|---
  |___|   |___|   |___|   |___|   |___|   |___|   |___|   |___|           |___|     |   
                                                                                   _|_  
                                                                                  |CX | 
----------------------------------------------------------------------------------|18 |-
                                                                                  |___|
```
Next, we create a noise model with some `NoiseInstruction`s. Here we only consider `MeasurementNoise`.


```python
from quri_parts.circuit.noise import BitFlipNoise, MeasurementNoise, NoiseModel

noise_model = NoiseModel([
    MeasurementNoise([BitFlipNoise(0.01)])
])
```

This noise model introduces bit flip error during the measurement.

### Readout error mitigation and peformance

Here, we explicitly show how to build an estimator that performs readout error mitigation. In this simple example, we will compare the performance of a [sampling estimator](../../sampling_estimation/index.md) that runs readout error mitigation with that of noiseless and noisy esimators. We first prepare an operator for this purpose.


```python
from quri_parts.core.operator import Operator, pauli_label, PAULI_IDENTITY
op = Operator({
    pauli_label("Z0"): 0.25,
    pauli_label("Z1 Z2"): 2.0,
    pauli_label("X1 X2"): 0.5,
    pauli_label("Z1 Y2"): 1.0,
    pauli_label("X1 Y2"): 2.0,
    PAULI_IDENTITY: 3.0,
})
```

Now, we demonstrate how to construct a readout error mitigation sampling estimator. As we shown in the [sampling estimator tutorial](../../sampling_estimation/index.md), the creation of a sampling estimator would require a concurrent sampler. In the case of a readout error mitigation sampling estimator, we will build a sampler that constructs a filter matrix and performs the mitigation scheme internally. This sampler is created with the `create_readout_mitigation_concurrent_sampler` function provided in `quri_parts.algo`, which we show below.


```python
from quri_parts.qulacs.estimator import create_qulacs_vector_estimator, create_qulacs_density_matrix_estimator
from quri_parts.qulacs.sampler import create_qulacs_density_matrix_concurrent_sampler
from quri_parts.core.estimator.sampling import create_sampling_estimator
from quri_parts.core.estimator import QuantumEstimator
from quri_parts.core.sampling.shots_allocator import create_equipartition_shots_allocator
from quri_parts.core.measurement import bitwise_commuting_pauli_measurement
from quri_parts.core.state import CircuitQuantumState, quantum_state
from quri_parts.algo.mitigation.readout_mitigation import create_readout_mitigation_concurrent_sampler

def get_readout_mit_estimator(qubit_count: int, shots: int=int(1e6)) -> QuantumEstimator[CircuitQuantumState]:
    noisy_concurrent_sampler = create_qulacs_density_matrix_concurrent_sampler(noise_model)
    allocator = create_equipartition_shots_allocator()

    readout_mit_concurrent_sampler = create_readout_mitigation_concurrent_sampler(
        qubit_count,
        noisy_concurrent_sampler,  # takes in a noisy concurrent sampler
        shots
    )
    return create_sampling_estimator(
        shots,
        readout_mit_concurrent_sampler,
        bitwise_commuting_pauli_measurement,
        allocator
    )

# Returns the estimator
readout_mit_estimator = get_readout_mit_estimator(qubit_count=3)
```

Now, we have a sampling estimator that performs readout error mitigation, which we compare its estimation result with that of a noiseless estimator and a noisy estimator.


```python
exact_estimator = create_qulacs_vector_estimator()
noisy_estimator = create_qulacs_density_matrix_estimator(noise_model)

state = quantum_state(3, circuit=circuit)

print("Noiseless estimate:", exact_estimator(op, state).value)
print("Noisy estimate:", noisy_estimator(op, state).value)
print("Readout mitigation estimate:", readout_mit_estimator(op, state).value)
```
```output
Noiseless estimate: (4.999999999999998+0j)
Noisy estimate: (4.920800000000014+0j)
Readout mitigation estimate: 5.000453637035886
```
We can see that the noise is indeed tamed and we obtain a better estimation of the expectation value compared to noisy estimation.

## Building the readout error mitigation sampler step by step

Now, we start to explain all the steps necessary to construct a readout error mitigation (concurrent) sampler. This envolves:

- Create a filter matrix from a noisy sampler.
- Apply the filter matrix to sampling counts produced by a noisy sampler.


### Sampling simulation with Qulacs

First, we create a noisy `Sampler` and execute the sampling without error mitigation.


```python
from quri_parts.qulacs.sampler import create_qulacs_density_matrix_sampler

sampler = create_qulacs_density_matrix_sampler(noise_model)
counts = sampler(circuit, shots=10000)
counts
```



```output
Counter({7: 4863, 0: 4834, 2: 65, 5: 51, 4: 50, 6: 47, 3: 46, 1: 44})
```


### Create filter matrix

We can use `create_filter_matrix` to create filter matrix.


```python
from quri_parts.algo.mitigation.readout_mitigation import create_filter_matrix
from quri_parts.qulacs.sampler import create_qulacs_density_matrix_concurrent_sampler

concurernt_sampler = create_qulacs_density_matrix_concurrent_sampler(noise_model)

filter_matrix = create_filter_matrix(qubit_count, concurernt_sampler, shots=1000000)
filter_matrix.round(5)
```



```output
array([[ 1.03103e+00, -1.04100e-02, -1.04900e-02,  1.10000e-04,
        -1.04900e-02,  1.10000e-04,  1.20000e-04, -0.00000e+00],
       [-1.05900e-02,  1.03113e+00,  1.10000e-04, -1.03100e-02,
         1.10000e-04, -1.04200e-02, -0.00000e+00,  1.10000e-04],
       [-1.04000e-02,  1.10000e-04,  1.03085e+00, -1.03400e-02,
         1.30000e-04, -0.00000e+00, -1.03600e-02,  1.00000e-04],
       [ 1.00000e-04, -1.05300e-02, -1.03200e-02,  1.03066e+00,
        -0.00000e+00,  1.10000e-04,  9.00000e-05, -1.03900e-02],
       [-1.03300e-02,  1.00000e-04,  1.10000e-04, -0.00000e+00,
         1.03079e+00, -1.04600e-02, -1.04400e-02,  1.00000e-04],
       [ 9.00000e-05, -1.05000e-02, -0.00000e+00,  1.10000e-04,
        -1.02300e-02,  1.03085e+00,  1.10000e-04, -1.05700e-02],
       [ 1.00000e-04, -0.00000e+00, -1.03600e-02,  1.10000e-04,
        -1.04100e-02,  1.10000e-04,  1.03075e+00, -1.04000e-02],
       [ 0.00000e+00,  1.10000e-04,  9.00000e-05, -1.03300e-02,
         1.10000e-04, -1.03000e-02, -1.02700e-02,  1.03106e+00]])
```


### Execute readout error mitigation

Now we can get error-mitigated counts by calling `readout_mitigation`.


```python
from quri_parts.algo.mitigation.readout_mitigation import readout_mitigation

mitigated_counts = readout_mitigation([counts], filter_matrix)
next(mitigated_counts)
```



```output
{0: 4982.354874161398,
 2: 16.273889374795427,
 4: 1.078239298564391,
 5: 0.643375217896129,
 7: 5012.556330314117}
```


### Create readout error mitigation sampler

We can also create a `ConcurrentSampler` that samples from noisy circuit and performs readout error mitigation behind the scenes.


```python
from quri_parts.algo.mitigation.readout_mitigation import (
    create_readout_mitigation_concurrent_sampler,
    create_readout_mitigation_sampler,
)
from quri_parts.qulacs.sampler import create_qulacs_density_matrix_sampler

# Create a ConcurrentSampler
rem_concurrent_sampler = create_readout_mitigation_concurrent_sampler(
    qubit_count, concurernt_sampler, shots=1000000
)

# You can also create a Sampler
# rem_sampler = create_readout_mitigation_sampler(
#     qubit_count, concurernt_sampler, shots=1000000
# )
# mitigated_counts = rem_sampler(circuit, 10000)
# print(mitigated_counts)

mitigated_counts_concurrent = rem_concurrent_sampler([(circuit, 10000)])
next(mitigated_counts_concurrent)

```



```output
{0: 4896.207916523141,
 1: 4.067266119004306,
 2: 3.9667263068103784,
 4: 3.201441039441285,
 5: 9.28954616971044,
 6: 0.8907393987065273,
 7: 5091.235819215903}
```

