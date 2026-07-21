# Generate a Bell State & Visualize Sampling Results

## Introduction

A **Bell state** is one of four maximally entangled two-qubit states, and in the two-qubit **Hilbert** space form an orthonormal basis. In quantum mechanics, a **Hilbert** space is a complete vector space equipped with an inner product. Each quantum state (like ∣0⟩, ∣1⟩, or a superposition) is represented as a vector in this space, and quantum operations are represented as linear transformations (matrices) acting on these vectors. Mathematically, the four Bell states are given by:

$$
\begin{gathered}
|\Phi^+\rangle = \frac{1}{\sqrt{2}} (|00\rangle + |11\rangle) \\
|\Phi^-\rangle = \frac{1}{\sqrt{2}} (|00\rangle - |11\rangle) \\
|\Psi^+\rangle = \frac{1}{\sqrt{2}} (|01\rangle + |10\rangle) \\
|\Psi^-\rangle = \frac{1}{\sqrt{2}} (|01\rangle - |10\rangle)
\end{gathered}
$$

Each Bell state represents a maximally entangled pair of qubits, meaning measurement outcomes on one qubit are perfectly correlated (or anticorrelated) with those of the other.

In the following example notebook, a two qubit quantum circuit is represented and the results visulised. 


```python
# %pip install "quri-parts[braket,cirq,qiskit,qulacs,tket]"
```

## Building a Simple Quantum Circuit

In this section, we construct a two-qubit quantum circuit that generates the Bell state. 

$$
|\Phi^+\rangle = \frac{1}{\sqrt{2}} (|00\rangle + |11\rangle)
$$

The core classes from the `quri_parts` library are imported, which provides tools for building and simulating quantum circuits with backends.



```python
from quri_parts.circuit import QuantumCircuit
from quri_parts.qulacs.sampler import create_qulacs_vector_sampler
from quri_parts.circuit.utils.circuit_drawer import draw_circuit
import matplotlib.pyplot as plt
from collections import Counter
```

A quantum circuit object with two qubits is initialized, where each qubit starts in the ∣0⟩ state.


```python
n_qubits = 2
circuit = QuantumCircuit(n_qubits)  
```

A Hadamard gate (H) on the first qubit creates a superposition. Then a CNOT gate entangles the two qubits, using the first as the control and the second as the target. Together, these operations produce the Bell state ∣Φ+⟩. As shown, the circuit can easily be visulised.


```python
circuit.add_H_gate(0)
circuit.add_CNOT_gate(0, 1)
draw_circuit(circuit)
```

The number of times the circuit should be executed (or sampled) is then specified. Here, `shots` represents one measurement of the two-qubit system after running the circuit. The more shots we take, the more accurately we can estimate the underlying probability distribution. In the following example, the qulacs vector sampler is a simulator backend from the QuriParts library. It uses state-vector simulation (as opposed to hardware execution) to compute measurement outcomes with high precision.


```python
# Create a sampler and perform sampling
shots = 1000
sampler = create_qulacs_vector_sampler()
sampling_result = sampler(circuit, shots=shots)
print(sampling_result)
```

This simple two-step circuit is the canonical example of quantum entanglement, forming the foundation for many quantum algorithms and communication protocols.

## Visualization of Sampling Results
A visulisation of the sampling results is projected onto a histogram. 


```python
def visualize_sampling_result(sampling_results, total_shots=1000, options="partial"):
    """
    Visualizes the sampling results by plotting a bar graph of the probabilities.

    Args:
        sampling_results (dict): A dictionary containing the sampling results, where the keys are the states and the values are the counts.
        total_shots (int, optional): The total number of shots. Defaults to 1000.
        options (str, optional): The options for processing the sampling results. Defaults to "partial".
            - "partial": Only consider the states present in the sampling results.
            - "complete": Consider all possible states and set the count to 0 if not present.
    """
    if options == "complete":
        # All possible states (0 to 2^n_qubits) with counts from sampling_results or 0 if not present
        all_states = set(range(2**n_qubits))
        fixed_results = {state: sampling_results.get(state, 0) for state in all_states}

        # Convert to Counter object for easier processing
        sampling_results = Counter(fixed_results)

    # Convert keys to binary representation for visualization
    binary_keys = [f'{key:0{n_qubits}b}' for key in sampling_results.keys()]

    # Calculate probabilities by dividing counts by the total shots
    probabilities = [count / total_shots for count in sampling_results.values()]

    # Plot the bar graph
    plt.bar(binary_keys, probabilities, color='crimson')

    # Add labels and title
    plt.xlabel('State (Binary Representation)')
    plt.ylabel('Probability')
    plt.title('Sampling Result')

    # Display the graph
    plt.show()

visualize_sampling_result(sampling_result, total_shots=shots, options="complete")
```

This result tells us that, out of 1000 measurements:

- About half of the outcomes were `00`
- About half were `11`
- No (or almost no) outcomes were `01` or `10`

This distribution matches the expected behavior of the **Bell state** earlier prepared. Here, the two qubits are **entangled** (their states are perfectly correlated). When we measure one qubit and find it in state `0`, the other is found in `0`; when one is measured as `1`, the other is `1`. Importantly, neither qubit individually has a definite state before measurement. 


### Other Bell States

To quickly prepare and visulise Bell states, the following function assembles the quantum circuits.


```python
def bell_state_circuit(label: str):
    """Return a circuit preparing one of the four Bell states."""
    qc = QuantumCircuit(2)
    qc.add_H_gate(0)
    qc.add_CNOT_gate(0, 1)

    if label == "Phi-":
        qc.add_Z_gate(0)  # Flip phase of |11⟩ component
    elif label == "Psi+":
        qc.add_X_gate(1)  # Flip target qubit to swap |00⟩→|01⟩, |11⟩→|10⟩
    elif label == "Psi-":
        qc.add_X_gate(1)
        qc.add_Z_gate(0)
    elif label != "Phi+":
        raise ValueError("label must be one of 'Phi+', 'Phi-', 'Psi+', 'Psi-'")

    return qc

# Example: create and draw each circuit
for label in ["Phi+", "Phi-", "Psi+", "Psi-"]:
    print(f"{label} circuit:")
    draw_circuit(bell_state_circuit(label))
    print("\n")

```

These quantum circuits can then be sampled and their results visulised. 
