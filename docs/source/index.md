# Welcome to QURI SDK

QURI SDK is a Python toolkit for quantum algorithm research and development.
Write an algorithm once, then simulate it, run it on real quantum devices, and
estimate the resources it needs on different quantum architectures, all with the
same code, without having to learn the details of each backend.

::::{grid} 1
:gutter: 3

:::{grid-item-card} 🚀 New to QURI SDK? Start here
:link: quick_start
:link-type: doc
:class-card: landing-card card-get-started

Install the SDK, build your first quantum circuit, and sample it, then take
your first resource estimate on a virtual machine.
:::

::::

## Find what you need

The documentation is organized by what you are trying to do:

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} 📘 Tutorials: learn the SDK
:link: docs/tutorials/index
:link-type: doc
:class-card: landing-card card-tutorials

Hands-on lessons, starting from the basics: per-library tutorials for QURI
Parts, QURI Algo and QURI VM, from your first circuit to advanced topics.
:::

:::{grid-item-card} 🛠️ How-to guides: get things done
:link: docs/howto/index
:link-type: doc
:class-card: landing-card card-howto

Goal-oriented guides by task: run and simulate circuits, mitigate errors,
compile circuits, and apply algorithms.
:::

:::{grid-item-card} 📖 Reference: look things up
:link: docs/reference/index
:link-type: doc
:class-card: landing-card card-reference

Consult while you work: implementations of textbook quantum algorithms, the
ecosystem, the API reference, and release notes.
:::

:::{grid-item-card} 💡 Concepts: understand the design
:link: docs/concepts/index
:link-type: doc
:class-card: landing-card card-concepts

Why QURI SDK is three libraries, how they fit together, and which one to
reach for.
:::

::::

## The three libraries

- **QURI Parts** provides the building blocks: circuits, states, operators,
  samplers and estimators. Most users start here.
- **QURI Algo** works at the algorithm level, with platform-independent
  algorithm definitions including Early-FTQC algorithms.
- **QURI VM** evaluates how an algorithm performs on a given architecture and
  device, covering circuit transpilation and quantum resource estimation.

They compose into one workflow: build with Parts, assemble algorithms with Algo,
and evaluate them with VM. For the full picture, see the
[QURI SDK Architecture](docs/concepts/overview/index.md).
