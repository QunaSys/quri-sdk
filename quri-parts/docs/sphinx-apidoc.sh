#!/bin/sh

set -e

pkgs="
    circuit
    core
    algo
    qulacs
    braket
    qiskit
    cirq
    chem
    openfermion
    stim
    openqasm
    quantinuum
    ionq
    itensor
    pyscf
    tket
    qsub
    tensornetwork
"

for pkg in $pkgs
do
    sphinx-apidoc -e -f --implicit-namespaces -o ./quri_parts/$pkg -M ../packages/$pkg/quri_parts
    # quri_parts.rst and modules.rst here are bare namespace-root stubs,
    # regenerated identically by every package invocation; nothing links to
    # them (reference.rst links the real per-package pages instead), and
    # leaving them in place makes every extra package's copy collide with
    # every other package's copy as a duplicate object description.
    rm -f ./quri_parts/$pkg/modules.rst ./quri_parts/$pkg/quri_parts.rst
done
