import copy
from typing import cast

import numpy as np

from quri_parts.qsub.lib.std import CNOT, H
from quri_parts.qsub.op import ParamOpDef, param_op
from quri_parts.qsub.opsub import ParamOpSubDef, param_opsub
from quri_parts.qsub.param import ArrayRef
from quri_parts.qsub.resolve import default_repository
from quri_parts.qsub.sub import SubBuilder


class TestArrayRefOpSub:
    def test_arrayop_float(self) -> None:
        length = 10
        xs = np.random.rand(length)

        class _F(ParamOpDef[ArrayRef[np.float64]]):
            name = "F"

            def qubit_count_fn(self, array: ArrayRef[np.float64]) -> int:
                assert np.array_equal(xs, array.array)
                return len(array.array) * 2

            def reg_count_fn(self, array: ArrayRef[np.float64]) -> int:
                assert np.array_equal(xs, array.array)
                return len(array.array) * 3

        F = param_op(_F)
        op = F(ArrayRef(xs))
        assert op.qubit_count == length * 2
        assert op.reg_count == length * 3
        assert np.array_equal(xs, cast(ArrayRef[np.float64], op.id.params[0]).array)

    def test_arrayop_copy(self) -> None:
        xs = np.random.rand(10)

        class _F(ParamOpDef[ArrayRef[np.float64]]):
            name = "F"

            def qubit_count_fn(self, array: ArrayRef[np.float64]) -> int:
                assert id(xs) == id(array.array)
                return len(array.array)

            def reg_count_fn(self, array: ArrayRef[np.float64]) -> int:
                assert id(xs) == id(array.array)
                return len(array.array)

        F = param_op(_F)
        op0 = F(ArrayRef(xs))
        assert id(xs) == id(cast(ArrayRef[np.float64], op0.id.params[0]).array)
        op1 = op0
        assert id(xs) == id(cast(ArrayRef[np.float64], op1.id.params[0]).array)
        op2 = copy.copy(op1)
        assert id(xs) == id(cast(ArrayRef[np.float64], op2.id.params[0]).array)
        op3 = copy.deepcopy(op2)
        assert id(xs) != id(cast(ArrayRef[np.float64], op3.id.params[0]).array)

    def test_array_eq_hash(self) -> None:
        xs = np.arange(10)
        ys = xs
        zs = np.arange(10)

        arr_a = ArrayRef(xs)
        arr_b = ArrayRef(ys)
        arr_c = ArrayRef(zs)
        arr_d = copy.copy(arr_a)
        arr_e = copy.deepcopy(arr_a)

        assert arr_a == arr_b
        assert arr_a != arr_c
        assert arr_a == arr_d
        assert arr_a != arr_e

        assert {arr_a, arr_b, arr_c, arr_d, arr_e} == {arr_a, arr_c, arr_e}

    def test_arrayopsub_resolve(self) -> None:
        xs = np.random.rand(10)
        ys = copy.deepcopy(xs)
        arr0 = ArrayRef(xs)
        arr1 = ArrayRef(ys)

        class _F(ParamOpSubDef[ArrayRef[np.float64]]):
            name = "F"
            qubit_count = 3

            def sub(self, builder: SubBuilder, array: ArrayRef[np.float64]) -> None:
                a, b, c = builder.qubits
                builder.add_op(H, (a,))
                builder.add_op(CNOT, (a, b))
                builder.add_op(CNOT, (b, c))

        F, FSub = param_opsub(_F)
        repo = default_repository()
        rslvr0 = repo.find_resolver(F(arr0))
        rslvr1 = repo.find_resolver(F(arr1))

        assert rslvr0 is not None
        assert rslvr0(F(arr0), repo) == FSub(arr0)
        assert rslvr1 is not None
        assert rslvr1(F(arr1), repo) == FSub(arr1)
