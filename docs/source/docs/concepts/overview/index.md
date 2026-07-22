# QURI SDK Architecture

QURI SDK is not a single library but three: QURI Parts, QURI Algo and QURI VM.
This page explains why the SDK is shaped this way, how the pieces relate, and
which one to reach for.

## Why three libraries?

Quantum algorithm research happens at very different levels of abstraction. Some
work is about circuits themselves: composing gates, preparing states, measuring
operators. Some work is about whole algorithms: time evolution, phase estimation,
variational loops. And in the (E)FTQC era, a growing share of the work is about
questions like *"how many logical qubits and how much runtime would this
algorithm need on that architecture?"*, questions that must be answered before
the hardware exists.

A single library serving all three needs would force every user through every
layer. QURI SDK instead separates the concerns:

- **QURI Parts** is the circuit level. It provides the building blocks
  (circuits, states, operators, samplers and estimators) with a design goal of
  hardware independence: the same code runs on simulators and real devices
  through interchangeable backends.
- **QURI Algo** is the algorithm level. It defines algorithms
  platform-independently and ships Early-FTQC algorithms ready to use, so that
  algorithm-level research does not have to wire circuits by hand.
- **QURI VM** is the evaluation level. It models quantum architectures and
  devices, transpiles circuits to error-corrected architectures, and estimates
  the resources an algorithm would consume: simulation and analysis for
  hardware that may not be available yet.

The reason for the split is the same as the reason compilers separate front ends
from back ends: each layer can evolve independently, and users enter at the
level that matches their problem.

## How the pieces relate

The layers compose rather than compete. An algorithm assembled with QURI Algo is
built out of QURI Parts components, and the result can be handed to QURI VM to
evaluate its cost on a chosen architecture. A typical research workflow touches
all three: build with Parts, assemble with Algo, evaluate with VM.

## Which should I use?

| I want to… | Start with |
|---|---|
| Build and simulate a quantum circuit | **QURI Parts** |
| Use a ready-made (Early-FTQC) algorithm | **QURI Algo** |
| Estimate resources / evaluate an algorithm on an architecture | **QURI VM** |

If you are unsure, start with QURI Parts: the other two build on its concepts.

## Related reading

- [Quick Start](../../../quick_start.md): install and run your first circuit.
- [Define your first qsub circuit](../../tutorials/quri-parts/advanced/qsub/basics.ipynb):
  the FTQC-oriented circuit abstraction.
- For quantum chemistry, a suggested reading order:
  [QURI Parts Tutorials](../../tutorials/quri-parts/index.md) →
  [Introduction to Quantum Chemistry](../../tutorials/quri-parts/quantum-chemistry/introduction/0_introduction.ipynb) →
  [Molecular Orbitals](../../tutorials/quri-parts/quantum-chemistry/mo/1_molecules.ipynb) →
  [Hamiltonian Generation](../../tutorials/quri-parts/quantum-chemistry/hamiltonian/hamiltonian/0_hamiltonian.ipynb).
