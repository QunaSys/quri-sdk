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

### [Generate a Bell State](Basics/bell/0_bell.ipynb)

Build a Bell state circuit and visualize sampling results.

### [Toffoli Truth Table by Sampling](Basics/toffoli/1_toffoli.ipynb)

Build a Toffoli (CCX) gate circuit and reconstruct its truth table by sampling.

### [Deutsch-Jozsa Algorithm](Intermediate/DJ/0._dj_algorithm.ipynb)

Determine whether an oracle is constant or balanced, deterministically.

### [Bernstein-Vazirani Algorithm](Intermediate/BV/1._bv_algorithm.ipynb)

Recover a secret bit string with a single oracle query.

### [Simon's Algorithm](Intermediate/simons/2._simons.ipynb)

Solve Simon's problem with a quantum computer.

### [Quantum Fourier Transform](Intermediate/QFT/0_qft.ipynb)

Implement the quantum Fourier transform and algorithms built on it.

### [Quantum Phase Estimation](Intermediate/QPE/0_qpe.ipynb)

Estimate the eigenvalue (phase) of a unitary operator for one of its eigenstates.

### [Surface Code and Quantum Error Correction](../quri-algo-vm/surface_code/4_surface_code.ipynb)

The planar surface code: definitions, terminology and a hands-on implementation.
