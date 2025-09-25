from numpy.testing import assert_almost_equal
from quri_parts.core.operator import PAULI_IDENTITY, Operator, pauli_label
from quri_parts.qsub.compile import compile_sub
from quri_parts.qsub.eval import QURIPartsEvaluatorHooks
from quri_parts.qsub.evaluate import Evaluator
from quri_parts.qsub.primitive import AllBasicSet
from quri_parts.qsub.resolve import resolve_sub
from quri_parts.qulacs.estimator import create_qulacs_general_vector_estimator

from quri_algo.circuit.time_evolution.trotter_time_evo import (
    get_trotter_time_evolution_operator,
)
from quri_algo.core.cost_functions.hilbert_schmidt_test import HilbertSchmidtTest
from quri_algo.problem import QubitHamiltonian
from quri_algo.qsub.time_evolution.trotter_time_evo import TrotterTimeEvo


def test_trotter_time_evo_qsub() -> None:
    qp_generator = Evaluator(QURIPartsEvaluatorHooks())
    hs_test = HilbertSchmidtTest(estimator=create_qulacs_general_vector_estimator())

    qubit_count = 2
    operator = Operator(
        {pauli_label("X0 X1"): 2, pauli_label("Y0 Y1"): 3, PAULI_IDENTITY: 5}
    )
    h = QubitHamiltonian(qubit_count, operator)
    t = 1.0

    n_trotter = 1
    order = 1

    te_sub = resolve_sub(TrotterTimeEvo(h, t, n_trotter, order))
    assert te_sub is not None
    te_msub = compile_sub(te_sub, AllBasicSet)
    te_qp = qp_generator.run(te_msub).freeze()

    te_parametric = get_trotter_time_evolution_operator(
        operator, qubit_count, n_trotter, order
    )
    te_target = te_parametric.bind_parameters([t])

    assert_almost_equal(hs_test(te_target, te_qp).value.real, 0.0)

    n_trotter = 2
    order = 1

    te_sub = resolve_sub(TrotterTimeEvo(h, t, n_trotter, order))
    assert te_sub is not None
    te_msub = compile_sub(te_sub, AllBasicSet)
    te_qp = qp_generator.run(te_msub).freeze()

    te_parametric = get_trotter_time_evolution_operator(
        operator, qubit_count, n_trotter, order
    )
    te_target = te_parametric.bind_parameters([t])

    assert_almost_equal(hs_test(te_target, te_qp).value.real, 0.0)

    n_trotter = 1
    order = 2

    te_sub = resolve_sub(TrotterTimeEvo(h, t, n_trotter, order))
    assert te_sub is not None
    te_msub = compile_sub(te_sub, AllBasicSet)
    te_qp = qp_generator.run(te_msub).freeze()

    te_parametric = get_trotter_time_evolution_operator(
        operator, qubit_count, n_trotter, order
    )
    te_target = te_parametric.bind_parameters([t])

    assert_almost_equal(hs_test(te_target, te_qp).value.real, 0.0)

    n_trotter = 2
    order = 2

    te_sub = resolve_sub(TrotterTimeEvo(h, t, n_trotter, order))
    assert te_sub is not None
    te_msub = compile_sub(te_sub, AllBasicSet)
    te_qp = qp_generator.run(te_msub).freeze()

    te_parametric = get_trotter_time_evolution_operator(
        operator, qubit_count, n_trotter, order
    )
    te_target = te_parametric.bind_parameters([t])

    assert_almost_equal(hs_test(te_target, te_qp).value.real, 0.0)
