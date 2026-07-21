<div class="alert alert-info">
    This notebook is no longer being maintained and we cannot guarantee that it will run
</div>

# Sampling on Qiskit's Real Quantum Computers

Here we introduce the Qiskit backends. For IBM devices, we provide 2 sampling backends:

- `QiskitSamplingBackend` : For the IBMQ backends
- `QiskitRuntimeSamplingBackend`: For the devices provided by the `QiskitRuntimeService`

which we provide further details on how to use them.

## Prerequisite


This section requires topics described in the previous sections ([Sampler](../../sampler/index.md), [Sampling Estimation](../../sampling_estimation/index.md) and [Sampling Backend](../sampling_backends/index.md)), so you need to read them before this section.

QURI Parts modules used in this tutorial: `quri-parts-circuit`, `quri-parts-core` and `quri-parts-qiskit`. You can install them as follows:


```ipython3
!pip install "quri-parts[qiskit]"
```

## QiskitSamplingBackend and Sampler

For using an IBMQ backend real device, you would need to activate your account first, then you may pick a device you prefer listed by the `all_device` variable defined below:


```python
from qiskit import IBMQ

provider = IBMQ.enable_account("Please input your token")
all_devices = provider.backends()
print(all_devices)
```

Then, a `QiskitSamplingBackend` may be created by passing in the device


```python
from quri_parts.qiskit.backend import QiskitSamplingBackend
backend = provider.get_backend("ibmq_qasm_simulator")
sampling_backend = QiskitSamplingBackend(backend=backend)
```

With the sampling backend we just created, we can run the exact codes as in the [Sampling Backend and Sampler](../sampling_backends/index.md#sampling-backend-and-sampler) section of the [Sampling Backend](../sampling_backends/index.md) tutorial.

## QiskitRuntimeSamplingBackend and Sampler

[Qiskit Runtime](https://qiskit.org/ecosystem/ibm-runtime/) is a new service provided by IBM to perform experiments on real devices. In this section, we demonstrate how to submit jobs to Qiskit Runtime via `QURI Parts`. We first prepare some circuit


```python
from quri_parts.circuit import QuantumCircuit

# construct quri-parts circuit
qp_circuit1 = QuantumCircuit(2)
qp_circuit1.add_H_gate(0)
qp_circuit1.add_H_gate(1)
qp_circuit1.add_RX_gate(0, 0.23)
qp_circuit1.add_RY_gate(1, -0.99)

qp_circuit2 = QuantumCircuit(2)
qp_circuit2.add_H_gate(0)
qp_circuit2.add_H_gate(1)
qp_circuit2.add_RX_gate(0, 1.23)
qp_circuit2.add_RY_gate(1, 4.56)

qp_circuit3 = QuantumCircuit(2)
qp_circuit3.add_H_gate(0)
qp_circuit3.add_H_gate(1)
qp_circuit3.add_RX_gate(0, 0.998)
qp_circuit3.add_RY_gate(1, 1.928)
```

For using the Qiskit Runtime service, you can create a `QiskitRuntimeSamplingBackend` by passing  in a `qiskit.providers.backend` and a `qiskit_ibm_runtime.QiskitRuntimeService` object. To see the list of all the devices supported by the Qiskit Runtime Service, you may run:


```python
from qiskit_ibm_runtime import QiskitRuntimeService

service = QiskitRuntimeService()

# Note: all available devices can be obtained by:
print(service.backends())
```
```output
[<IBMBackend('ibmq_qasm_simulator')>, <IBMBackend('ibmq_lima')>, <IBMBackend('simulator_statevector')>, <IBMBackend('ibm_nairobi')>, <IBMBackend('ibm_lagos')>, <IBMBackend('ibm_perth')>, <IBMBackend('ibmq_jakarta')>, <IBMBackend('ibmq_quito')>, <IBMBackend('ibmq_belem')>, <IBMBackend('simulator_extended_stabilizer')>, <IBMBackend('simulator_mps')>, <IBMBackend('simulator_stabilizer')>, <IBMBackend('ibmq_manila')>]
```
We are now ready to create a qiskit runtime sampling backend


```python
from quri_parts.qiskit.backend import QiskitRuntimeSamplingBackend

backend = service.backend("ibmq_qasm_simulator")
qiskit_runtime_sampling_backend = QiskitRuntimeSamplingBackend(
    backend=backend,
    service=service,
)
```

A sampler can be created as usual


```python
from quri_parts.core.sampling import create_sampler_from_sampling_backend

sampler = create_sampler_from_sampling_backend(qiskit_runtime_sampling_backend)

sampling_cnt_1 = sampler(qp_circuit1, 1000)
sampling_cnt_2 = sampler(qp_circuit2, 2000)
sampling_cnt_3 = sampler(qp_circuit3, 3000)
```

### Grouping multiple sampling jobs into a Session

In Qiskit Runtime Service, they provide `Session` objects that allows us to group jobs together. In the above example, a `Session` is created every time the sampler is called. In `QURI Parts`, we can group jobs into a single `Session` with the `QiskitRuntimeSamplingBackend` as well.


```python
with QiskitRuntimeSamplingBackend(backend=backend, service=service) as qiskit_runtime_sampling_backend:
	sampler = create_sampler_from_sampling_backend(qiskit_runtime_sampling_backend)
	sampling_cnt_1 = sampler(qp_circuit1, 1000)
	sampling_cnt_2 = sampler(qp_circuit2, 2000)
	sampling_cnt_3 = sampler(qp_circuit3, 3000)
```

### Billable Time Tracker

When executing jobs on real devices, it can be useful to keep track of the cost by tracking the billable time. In the `QiskitRuntimeSamplingBackend`, we provide the `total_time_limit` option that allows you to track the total billable time. If the total billable time exceeds the time limit, the backend will reject new job submissions and cancel all unfinished jobs. Let’s create a backend with run time limit of 100 seconds:


```python
TIME_LIMIT = 100

backend = service.backend("ibmq_qasm_simulator")

sampling_backend = QiskitRuntimeSamplingBackend(
	backend=backend,
	service=service,
	total_time_limit=TIME_LIMIT
)
```

When the total time limit is set, a `Tracker` object is created along with the sampling backend. You may access it by the `.tracker` attribute.


```python
tracker = sampling_backend.tracker
```

The total billable run time of jobs submitted by the backend is not tracked automatically. Instead, it is computed whenever any of the `total_run_time`, `running_jobs`, `finished_jobs` properties is accessed. As we have not submitted any jobs with the backend we just created, it should be 0 at the moment.


```python
tracker.total_run_time
```



```output
0.0
```


When we submit jobs with the backend, the jobs will be registered to the tracker for tracking. For example let’s submit 3 sampling jobs:


```python
sampling_job_1 = sampling_backend.sample(qp_circuit1, 10)
sampling_job_2 = sampling_backend.sample(qp_circuit2, 20)
sampling_job_3 = sampling_backend.sample(qp_circuit3, 30)
```

These jobs will be stored inside the tracker and can be accessed by the `.running_jobs` property if they are still being executed by the real device. If any of the job is finished, you may access them with the `.finished_jobs` property.


```python
print(tracker.running_jobs)
print(tracker.finished_jobs)
```
```output
[<quri_parts.qiskit.backend.primitive.QiskitRuntimeSamplingJob object at 0x17c66f1f0>, <quri_parts.qiskit.backend.primitive.QiskitRuntimeSamplingJob object at 0x17b24fdf0>, <quri_parts.qiskit.backend.primitive.QiskitRuntimeSamplingJob object at 0x17b24a3d0>]
[]
```
## Data Saving and Replaying

As we might want to perform different analysis using the same data generated by real devices, having a way to save and retrieve past experiment data can be useful. In this section, we explain how to save and replay past experiment data generated by Qiskit devices.

The data saving feature can be activated by setting the `save_data_while_sampling` to True. Both `QiskitSamplingBackend` and `QiskitRuntimeSamplingBackend` support this feature. Let’s use the local Aer simulator as an example.


```python
from quri_parts.qiskit.backend import QiskitSamplingBackend
from qiskit_aer import AerSimulator
from quri_parts.core.sampling import create_sampler_from_sampling_backend

sampling_backend = QiskitSamplingBackend(
	backend=AerSimulator(),
	save_data_while_sampling=True # activate data saving feature
)

sampler = create_sampler_from_sampling_backend(sampling_backend)

cnt1 = sampler(qp_circuit1, 100)
cnt2 = sampler(qp_circuit2, 200)
cnt3 = sampler(qp_circuit3, 300)

print(cnt1)
print(cnt2)
print(cnt3)
```
```output
{2: 3, 3: 5, 0: 56, 1: 36}
{3: 1, 2: 1, 0: 108, 1: 90}
{1: 7, 0: 9, 2: 147, 3: 137}
```
After performing sampling job like the above, we may save the sampling data into a json file:


```python
import json

with open('saved_sampling_job.json', 'w') as fp:
	json.dump(sampling_backend.jobs_json, fp)
```

The `jobs_json` property accessed above encodes all the past sampling jobs in the order they were submitted. Now, let’s load it back to the memory and replay with the `QiskitSavedDataSamplingBackend`.


```python
from quri_parts.qiskit.backend import QiskitSavedDataSamplingBackend

with open('saved_sampling_job.json', 'r') as fp:
	saved_data = json.load(fp)

replay_backend = QiskitSavedDataSamplingBackend(
	backend=AerSimulator(),
	saved_data=saved_data
)

replay_sampler = create_sampler_from_sampling_backend(replay_backend)

replay_cnt1 = replay_sampler(qp_circuit1, 100)
replay_cnt2 = replay_sampler(qp_circuit2, 200)
replay_cnt3 = replay_sampler(qp_circuit3, 300)

print(replay_cnt1)
print(replay_cnt2)
print(replay_cnt3)
```
```output
{2: 3, 3: 5, 0: 56, 1: 36}
{3: 1, 2: 1, 0: 108, 1: 90}
{1: 7, 0: 9, 2: 147, 3: 137}
```