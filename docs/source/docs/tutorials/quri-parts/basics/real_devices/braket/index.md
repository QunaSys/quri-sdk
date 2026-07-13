# Sampling on Braket's Real Quantum Computers

Here we introduce the Braket backends and some Braket specific features QURI Parts provide

## Prerequisite

This section requires topics described in previous sections([Samplers](../../sampler/index.md), [Sampling Estimation](../../sampling_estimation/index.md) and [Sampling Backends](../sampling_backends/index.md)), so you need to read them before this section.

In this section, we use [Amazon Braket](https://aws.amazon.com/braket/) as the platform with real quantum computers. In order to use Braket devices provided on AWS, you need to have an AWS account and enable Braket service. Please see [Amazon Braket Documentation](https://docs.aws.amazon.com/braket/index.html) for details. In this section, instead, we use the local simulator included in [Amazon Braket SDK](https://amazon-braket-sdk-python.readthedocs.io/en/latest/index.html), which does not require an AWS account. The Braket devices provided on AWS and the local simulator have the same interface, you can simply replace them each other.

QURI Parts modules used in this tutorial: `quri-parts-circuit`, `quri-parts-core` and `quri-parts-braket`. You can install them as follows:


```ipython3
!pip install "quri-parts[braket]"
```

## The BraketSamplingBackend

How to create a `SamplingBackend` object depends on the used backend. For Braket devices, you can create a `BraketSamplingBackend` by passing a `braket.devices.Device` object (provided by Amazon Braket SDK):


```python
from braket.aws import AwsDevice
from braket.devices import LocalSimulator

# A device for QPU provided on AWS
# device = AwsDevice("arn:aws:braket:us-west-1::device/qpu/rigetti/Aspen-M-2")

# A device for the local simulator
device = LocalSimulator()
```


```python
from quri_parts.braket.backend import BraketSamplingBackend

# Create a SamplingBackend with the device
backend = BraketSamplingBackend(device)
```

With the sampling backend we just created, we can run the exact codes as in the [Sampling Backend and Sampler](../sampling_backends/index.md#sampling-backend-and-sampler) section of the Sampling backend tutorial.

## Qubit mapping

Here, we explain some details you need to know when you use the devices provided by Braket. Following the code in the [Sampling Backend](../sampling_backends/index.md) tutorial, we consider the following qubit mapping sampling.


```python
from numpy import pi
from quri_parts.circuit import QuantumCircuit
from quri_parts.core.sampling import create_sampler_from_sampling_backend

circuit = QuantumCircuit(4)
circuit.add_X_gate(0)
circuit.add_H_gate(1)
circuit.add_Y_gate(2)
circuit.add_CNOT_gate(1, 2)
circuit.add_RX_gate(3, pi/4)

backend = BraketSamplingBackend(device, qubit_mapping={0: 3, 1: 2, 2: 0, 3: 1})
sampler = create_sampler_from_sampling_backend(backend)
sampling_result = sampler(circuit, 1000)
print(sampling_result)
```
```output
{5: 427, 11: 84, 3: 434, 13: 55}
```
The result looks similar to one with no qubit mapping, since the measurement result from the device is mapped backward so that it is interpreted in terms of the original qubit indices.

<div class="alert alert-info">
    You may notice that the above mapping is a permutation of the original qubit indices and device qubits with indices larger than 3 are not involved. The reason for choosing such a mapping is to avoid an error of <code>LocalSimulator</code>: the <code>LocalSimulator</code> does not accept non-contiguous qubit indices. On the other hand, the qubit mapping feature of the <code>SamplingBackend</code> accepts such a mapping, as shown below.
</div>

When you apply qubit mapping to devices provided on AWS, you will need to [enable manual qubit allocation by passing disable_qubit_rewiring=True](https://docs.aws.amazon.com/braket/latest/developerguide/braket-constructing-circuit.html#manual-qubit-allocation) to the device. You can specify such an argument (i.e. keyword arguments for `run` method of a `braket.devices.Device` object) via `run_kwargs` argument of the `BraketSamplingBackend` object:


```python
# Commented out because it requires an access to a real device on AWS

# device = AwsDevice("arn:aws:braket:us-west-1::device/qpu/rigetti/Aspen-M-2")
# backend = BraketSamplingBackend(
#     device,
#     qubit_mapping={0: 10, 1: 13, 2: 17, 3: 21},
#     run_kwargs={"disable_qubit_rewiring": True},
# )
# sampler = create_sampler_from_sampling_backend(backend)
# sampling_result = sampler(circuit, 1000)
# print(sampling_result)
```

## Circuit transpilation before execution

The transpilation performed by default depends on the backend; in the case of `BraketSamplingBackend`, it uses `quri_parts.braket.circuit.BraketSetTranspiler` for all devices, and also performs some device-specific transpilation defined in `quri_parts.braket.backend.transpiler`. It is possible to change the former one (device-independent transpilation) by supplying `circuit_transpiler` argument to `BraketSamplingBackend`.

## Data Saving and Replaying

As we might want to perform different analysis using the same data generated by real devices, having a way to save and retrieve past experiment data can be useful. In this section, we explain how to save and replay past experiment data generated by Qiskit devices.

The data saving feature can be activated by setting the `save_data_while_sampling` to True. Both `BraketSamplingBackend` support this feature. Let’s use the local Aer simulator as an example.


```python
from quri_parts.braket.backend import BraketSamplingBackend
from braket.devices import LocalSimulator
from quri_parts.core.sampling import create_sampler_from_sampling_backend

circuit_1 = QuantumCircuit(4)
circuit_1.add_X_gate(0)
circuit_1.add_H_gate(1)
circuit_1.add_Y_gate(2)
circuit_1.add_CNOT_gate(1, 2)
circuit_1.add_RX_gate(3, pi/4)


circuit_2 = QuantumCircuit(4)
circuit_2.add_X_gate(0)
circuit_2.add_H_gate(1)
circuit_2.add_Y_gate(2)
circuit_2.add_CNOT_gate(1, 2)
circuit_2.add_RX_gate(3, pi/8)


sampling_backend = BraketSamplingBackend(
	device=LocalSimulator(),
	save_data_while_sampling=True # activate data saving feature
)

sampler = create_sampler_from_sampling_backend(sampling_backend)

cnt1 = sampler(circuit_1, 100)
cnt2 = sampler(circuit_2, 200)

print(cnt1)
print(cnt2)
```
```output
Counter({3: 47, 5: 35, 11: 11, 13: 7})
Counter({3: 110, 5: 81, 11: 6, 13: 3})
```
After performing sampling job like the above, we may save the sampling data into a json file:


```python
import json

with open('saved_sampling_job.json', 'w') as fp:
	json.dump(sampling_backend.jobs_json, fp)
```

The `jobs_json` property accessed above encodes all the past sampling jobs in the order they were submitted. Now, let’s load it back to the memory and replay with the `BraketSavedDataSamplingBackend`.


```python
from quri_parts.braket.backend import BraketSavedDataSamplingBackend

with open('saved_sampling_job.json', 'r') as fp:
	saved_data = json.load(fp)

replay_backend = BraketSavedDataSamplingBackend(
	device=LocalSimulator(),
	saved_data=saved_data
)

replay_sampler = create_sampler_from_sampling_backend(replay_backend)

replay_cnt1 = replay_sampler(circuit_1, 100)
replay_cnt2 = replay_sampler(circuit_2, 200)

print(replay_cnt1)
print(replay_cnt2)
```
```output
{5: 35, 3: 47, 13: 7, 11: 11}
{3: 110, 5: 81, 11: 6, 13: 3}
```