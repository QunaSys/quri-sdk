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
    honeywell
    ionq
    pyscf
"

for pkg in $pkgs
do
    sphinx-apidoc -e -f --implicit-namespaces -o ./quri_parts/$pkg -M ../packages/$pkg/quri_parts
done
