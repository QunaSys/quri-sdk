# Introduction to QURI VM

QURI VM is a mechanism that abstracts the architecture and devices of FTQC and NISQ quantum computers. This allows you to create quantum algorithms in a way that is independent of the architecture and devices, and then evaluate, optimize, simulate, and execute them on various architecture and devices.

Here, we will look at the basic usage of VMs. A VM instance holds information about the target architecture and device. By using VMs with different settings, you can perform operations on various architectures and devices.

As a preliminary step, let's define a quantum circuit, quantum state, and operator using QURI Parts as the components of the quantum algorithm that the VM will execute.


```python
from math import pi
from quri_parts.circuit import QuantumCircuit
from quri_parts.core.operator import pauli_label
from quri_parts.core.state import quantum_state
from quri_parts.circuit.utils.circuit_drawer import draw_circuit

circuit = QuantumCircuit(2)
circuit.add_H_gate(0)
circuit.add_RX_gate(1, pi / 3)
circuit.add_CNOT_gate(0, 1)
draw_circuit(circuit)

state = quantum_state(2, circuit=circuit)

op = pauli_label("Z0 Z1")
```
```output
   ___          
  | H |         
--|0  |-----●---
  |___|     |   
   ___     _|_  
  |RX |   |CX | 
--|1  |---|2  |-
  |___|   |___|
```
## Create a VM instance

The simplest way to create a VM instance is to create it without giving any settings. In this case, since no information about the architecture or devices is given, this VM instance can only be used for ideal (noiseless) simulations at the logical quantum circuit level. Qulacs is used as the default quantum circuit simulator.


```python
from quri_vm.vm import VM

# Here we call it an "abstrct VM" as it does not contain any information of architecture or devices.
abstract_vm = VM()
```

## Using a VM instance: sampling, estimation and analysis

VM provides APIs for sampling of quantum circuits and estimation of operator expectation values. These have a common interface with QURI Parts.
- Sampling: The `sample()` method provides the `Sampler` interface of QURI Parts.
- Expectation value estimation: The `estimate()` method provides the `QuantumEstimator` interface of QURI Parts.


```python

# Ideal (noiseless) sampling and estimation with Qulacs

ideal_samples = abstract_vm.sample(circuit, shots=1000)
print(ideal_samples)

ideal_estimate = abstract_vm.estimate(op, state)
print(ideal_estimate)
```
```output
Counter({0: 389, 3: 346, 2: 142, 1: 123})
_Estimate(value=(0.5+0j), error=0.0)
```
One of the best practices when writing algorithms is to write them as functions that take a VM as an argument. This allows you to separate the VM instance from the algorithm, and you can switch VM instances without changing the algorithm code.


```python
def my_algorithm(vm: VM):
    samples = vm.sample(circuit, 1000)
    estimate = vm.estimate(op, state)
    return samples, estimate

ideal_result = my_algorithm(abstract_vm)
print(ideal_result)
```
```output
(Counter({3: 386, 0: 365, 2: 132, 1: 117}), _Estimate(value=(0.5+0j), error=0.0))
```
VM can evaluate the performance of quantum circuits on a given architecture/device. The `abstract_vm` we just created does not have information about the architecture/device, so you can only get general information such as the number of qubits, gates, and depth. We will look at an example of evaluating performance on an architecture/device later.


```python
from pprint import pprint

ideal_analysis = abstract_vm.analyze(circuit)
pprint(ideal_analysis)
```
```output
AnalyzeResult(lowering_level=<LoweringLevel.LogicalCircuit: 0>,
              qubit_count=2,
              gate_count=3,
              depth=2,
              latency=None,
              fidelity=1.0)
```
## Performance Evaluation on the Early-FTQC Architecture

Using a VM, you can simulate and analyze quantum algorithms on the early-FTQC architecture. The following are some of the elements defined by the architecture that affect algorithm analysis and simulation.
- The native gate set supported by the architecture
- The quantum error correction code (type of code, code distance)
- Quantum error correction (QEC) cycle time
- Physical error rate of the device

As an example, let's specify the STAR architecture, which is one of the leading early-FTQC architectures. First, we will create a VM without specifying the physical error rate. In this case, we cannot perform fidelity analysis or noisy simulations, but we can evaluate the execution time of the quantum circuit. When performing analysis or simulation, the quantum circuit is transpiled to a native gate set, but since no noise is introduced, there is no effect on the simulation results.


```python
from quri_parts.backend.devices import star_device
from quri_parts.backend.units import TimeUnit, TimeValue

ideal_star_vm = VM.from_device_prop(star_device.generate_device_property(
    qubit_count=16,
    code_distance=7,
    qec_cycle=TimeValue(value=1.0, unit=TimeUnit.MICROSECOND),
))

ideal_star_result = my_algorithm(ideal_star_vm)
print(ideal_star_result)

ideal_star_analysis = ideal_star_vm.analyze(circuit)
pprint(ideal_star_analysis)
```
```output
(Counter({3: 386, 0: 369, 2: 127, 1: 118}), _Estimate(value=(0.5000000000000007+0j), error=0.0))
AnalyzeResult(lowering_level=<LoweringLevel.ArchLogicalCircuit: 1>,
              qubit_count=2,
              gate_count=5,
              depth=4,
              latency=TimeValue(value=84000.0, unit=<TimeUnit.NANOSECOND>),
              fidelity=1.0)
```
The result of the analysis shows that it takes 84 microseconds to execute the entire circuit.

Now, let's try running it with a specified physical error rate. In this case, the fidelity of the circuit can be evaluated through analysis. The simulation also uses a noise model defined by the characteristics of the architecture.


```python
noisy_star_vm = VM.from_device_prop(star_device.generate_device_property(
    qubit_count=16,
    code_distance=7,
    qec_cycle=TimeValue(value=1.0, unit=TimeUnit.MICROSECOND),
    physical_error_rate=1.0e-4,
))

noisy_star_result = my_algorithm(noisy_star_vm)
print(noisy_star_result)

noisy_star_analysis = noisy_star_vm.analyze(circuit)
pprint(noisy_star_analysis)
```
```output
(Counter({0: 373, 3: 362, 2: 135, 1: 130}), _Estimate(value=(0.4999733335111117+0j), error=0.0))
AnalyzeResult(lowering_level=<LoweringLevel.ArchLogicalCircuit: 1>,
              qubit_count=2,
              gate_count=5,
              depth=4,
              latency=TimeValue(value=84000.0, unit=<TimeUnit.NANOSECOND>),
              fidelity=0.9999341878196069)
```
## Available architectures and devices

The architecture and device settings provided by QURI VM are as follows.

### STAR: an early-FTQC architecture

The STAR architecture [1] is an early-FTQC architecture based on surface codes. We have already introduced how to create a VM that supports the STAR architecture above.

### Clifford+T: an FTQC architecture

The Clifford+T is an architecture where an arbitrary logical quantum gate is performed by decomposing it into Clifford gates and T gates. The Clifford+T architecture that the current QURI VM supports is based on surface codes, which is described in [2].


```python
from quri_parts.backend.devices import clifford_t_device

noisy_clifford_t_vm = VM.from_device_prop(clifford_t_device.generate_device_property(
    qubit_count=16,
    code_distance=7,
    qec_cycle=TimeValue(value=1.0, unit=TimeUnit.MICROSECOND),
    delta_sk=1.0e-4, # Specifies the precision of decomposition of rotation gates
    mode_block="compact", # Specifies the mode of block layout defined in [2]. compact, intermediate or fast.
    physical_error_rate=1.0e-4,
))

noisy_clifford_t_result = my_algorithm(noisy_clifford_t_vm)
print(noisy_clifford_t_result)

noisy_clifford_t_analysis = noisy_clifford_t_vm.analyze(circuit)
pprint(noisy_clifford_t_analysis)
```
```output
(Counter({3: 378, 0: 363, 2: 136, 1: 123}), _Estimate(value=(0.5+0j), error=0.0))
AnalyzeResult(lowering_level=<LoweringLevel.ArchLogicalCircuit: 1>,
              qubit_count=2,
              gate_count=5,
              depth=4,
              latency=TimeValue(value=2576000.0, unit=<TimeUnit.NANOSECOND>),
              fidelity=0.9989813634463354)
```
### Typical NISQ devices

You can also create a VM for a typical NISQ device by specifying device parameters. Currently supported devices are:

- A superconducting qubit device with square lattice topology
- A trapped ion qubit device with all-to-all connectivity

Please note that the noisy simulation is not available for those NISQ devices at this moment. The simulation is performed without applying the noise model.


```python
from quri_parts.circuit.topology import SquareLattice
from quri_parts.backend.devices import nisq_spcond_lattice, nisq_iontrap_device

# Superconducting NISQ device
nisq_spcond_vm = VM.from_device_prop(
    nisq_spcond_lattice.generate_device_property(
        lattice=SquareLattice(4, 4),
        native_gates_1q=("RZ", "SqrtX", "X"),
        native_gates_2q=("CNOT",),
        gate_error_1q=1e-3,
        gate_error_2q=1e-2,
        gate_error_meas=1e-2,
        gate_time_1q=TimeValue(60, TimeUnit.NANOSECOND),
        gate_time_2q=TimeValue(660, TimeUnit.NANOSECOND),
        gate_time_meas=TimeValue(1.4, TimeUnit.MICROSECOND),
    )
)

# The noise model is not applied for the NISQ device at the moment
nisq_spcond_result = my_algorithm(nisq_spcond_vm)
print(nisq_spcond_result)

nisq_spcond_analysis = nisq_spcond_vm.analyze(circuit)
pprint(nisq_spcond_analysis)

# Ion trap NISQ device
nisq_iontrap_vm = VM.from_device_prop(
    nisq_iontrap_device.generate_device_property(
        qubit_count=16,
        native_gates_1q=("RZ", "SqrtX", "X"),
        native_gates_2q=("CNOT",),
        gate_error_1q=1.53e-3,
        gate_error_2q=4e-2,
        gate_error_meas=2.5e-3,
        gate_time_1q=TimeValue(10, TimeUnit.MICROSECOND),
        gate_time_2q=TimeValue(200, TimeUnit.MICROSECOND),
        gate_time_meas=TimeValue(130, TimeUnit.MICROSECOND),
    )
)

# The noise model is not applied for the NISQ device at the moment
nisq_iontrap_result = my_algorithm(nisq_iontrap_vm)
print(nisq_iontrap_result)

nisq_iontrap_analysis = nisq_iontrap_vm.analyze(circuit)
pprint(nisq_iontrap_analysis)
```
```output
(Counter({3: 388, 0: 386, 2: 117, 1: 109}), _Estimate(value=(0.4643718016677503+0j), error=0.0))
AnalyzeResult(lowering_level=<LoweringLevel.ArchLogicalCircuit: 1>,
              qubit_count=2,
              gate_count=13,
              depth=8,
              latency=TimeValue(value=1080.0, unit=<TimeUnit.NANOSECOND>),
              fidelity=0.9781851226892669)
(Counter({0: 393, 3: 373, 2: 118, 1: 116}), _Estimate(value=(0.43911271039289523+0j), error=0.0))
AnalyzeResult(lowering_level=<LoweringLevel.ArchLogicalCircuit: 1>,
              qubit_count=2,
              gate_count=13,
              depth=8,
              latency=TimeValue(value=270000.0, unit=<TimeUnit.NANOSECOND>),
              fidelity=0.942521965592581)
```
## Summary

In this notebook, we explored the usage of QURI VM for abstracting the architecture and devices of quantum computers. We have emphasized the intended modular usage of QURI VM and how algorithms should be written to support using arbitrary VM instances. We have also showed how to instance and use VMs for resource estimation. In summary, by following this notebook you will have learned how to

- Create an abstract VM instance without any specific architecture or device settings for ideal (noiseless) simulations.
- Perform sampling and expectation value estimation using the `sample()` and `estimate()` methods provided by the VM.
- Write algorithms as functions that take a VM as an argument to separate the VM instance from the algorithm.
- Evaluating the performance of quantum circuits on different architectures and devices using the `analyze()` method.

By the end of this notebook, you should have a good understanding of how to use QURI VM to create, simulate, and analyze quantum algorithms on various quantum architectures and devices.

### Take-aways from this tutorial

Using what you have learned here, try running the noisy Clifford+T analysis with a range of code distances and error rates. Then plot the different outcomes to compare latency and fidelity using matplotlib.

## References

1. Partially Fault-Tolerant Quantum Computing Architecture with Error-Corrected Clifford Gates and Space-Time Efficient Analog Rotations. Akahoshi et al., PRX Quantum 5, 010337 (2024). doi:[10.1103/PRXQuantum.5.010337](https://link.aps.org/doi/10.1103/PRXQuantum.5.010337)
2. A Game of Surface Codes: Large-Scale Quantum Computing with Lattice Surgery. D Litinski, Quantum 3, 128 (2019). doi:[10.22331/q-2019-03-05-128](https://doi.org/10.22331/q-2019-03-05-128)
