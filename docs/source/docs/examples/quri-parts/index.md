# Quantum Algorithms

Complete implementations of textbook quantum algorithms and circuits, from Bell
states up to quantum phase estimation and the surface code. Consult them while
working and adapt them to your own code.

```{note}
Looking to apply an algorithm to a concrete problem (QSCI, statistical phase
estimation, and more)? See
[Applying Algorithms](../../howto/applying_algorithms/index.md) in the how-to
guides.
```

## Contents

### [Generate a Bell State](Basics/bell/index.md)

Build a Bell state circuit and visualize sampling results.

### [Toffoli Truth Table by Sampling](Basics/toffoli/index.md)

Build a Toffoli (CCX) gate circuit and reconstruct its truth table by sampling.

### [Deutsch-Jozsa Algorithm](Intermediate/DJ/index.md)

Determine whether an oracle is constant or balanced, deterministically.

### [Bernstein-Vazirani Algorithm](Intermediate/BV/index.md)

Recover a secret bit string with a single oracle query.

### [Simon's Algorithm](Intermediate/simons/index.md)

Solve Simon's problem with a quantum computer.

### [Quantum Fourier Transform](Intermediate/QFT/index.md)

Implement the quantum Fourier transform and algorithms built on it.

### [Quantum Phase Estimation](Intermediate/QPE/index.md)

Estimate the eigenvalue (phase) of a unitary operator for one of its eigenstates.

### [Surface Code and Quantum Error Correction](../quri-algo-vm/surface_code/4_surface_code.ipynb)

The planar surface code: definitions, terminology and a hands-on implementation.
