# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import cmath

from .pauli import PAULI_IDENTITY, PauliLabel, pauli_product


class Operator(dict[PauliLabel, complex]):
    """Operator represents the set of single-term operators as a
    dict[PauliLabel, coefficient].

    Coefficients can not only be real values but also complex values since Operator
    represents a general operator including non-Hermitian one.

    Example:
        >>> op = Operator({pauli_label("X0"): 0.1j})
        >>> op
        {PauliLabel({(0, <SinglePauli.X: 1>)}): 0.1j}
        >>> op[pauli_label("X1 Y2")] = 0.2
        >>> op
        {PauliLabel({(0, <SinglePauli.X: 1>)}): 0.1j, \
PauliLabel({(1, <SinglePauli.X: 1>), (2, <SinglePauli.Y: 2>)}): 0.2}
        >>> str(op)
        '0.1j*X0 + 0.2*X1 Y2'
        >>> for pauli, coef in op.items():
        ...     print(f"Pauli: {pauli}, coefficient: {coef}")
        ...
        Pauli: X0, coefficient: 0.1j
        Pauli: X1 Y2, coefficient: 0.2
        >>> op.constant
        0.0
        >>> op[PAULI_IDENTITY] = 2.0
        >>> op
        {PauliLabel({(0, <SinglePauli.X: 1>)}): 0.1j, \
PauliLabel({(1, <SinglePauli.X: 1>), (2, <SinglePauli.Y: 2>)}): 0.2, PauliLabel(): 2.0}
        >>> op.constant
        2.0
        >>> op.constant = 1.0
        >>> op.constant
        1.0

    You can add a single-term operator by `add_term()` method.
        >>> operator.add_term(PauliLabel({(1, 1)}), 1.0)
    By accessing by key, the coefficient is replaced with a new value.
        >>> operator = Operator({pauli_label("X0"): 0.1j})
        >>> op
        {PauliLabel({(0, <SinglePauli.X: 1>)}): 0.1j}
        >>> operator[pauli_label("X0")] = 0.1
        >>> op
        {PauliLabel({(0, <SinglePauli.X: 1>)}): 0.1}
    """

    def __str__(self) -> str:
        return " + ".join(
            [f"{coef}*{str(pauli_label)}" for pauli_label, coef in self.items()]
        )

    def __add__(self, other: object) -> "Operator":
        if not isinstance(other, Operator):
            return NotImplemented
        copied = self.copy()
        copied += other
        return copied

    def __iadd__(self, other: object) -> "Operator":
        if not isinstance(other, Operator):
            return NotImplemented
        for pauli_other, coef_other in other.items():
            self.add_term(pauli_other, coef_other)
        return self

    def __sub__(self, other: object) -> "Operator":
        if not isinstance(other, Operator):
            return NotImplemented
        copied = self.copy()
        copied -= other
        return copied

    def __isub__(self, other: object) -> "Operator":
        if not isinstance(other, Operator):
            return NotImplemented
        for pauli_other, coef_other in other.items():
            self.add_term(pauli_other, -1 * coef_other)
        return self

    def __mul__(self, other: object) -> "Operator":
        if isinstance(other, (int, float, complex)):
            return Operator({pauli: other * coef for pauli, coef in self.items()})
        elif isinstance(other, Operator):
            ret_op = Operator()
            for pauli, coef in self.items():
                for pauli_other, coef_other in other.items():
                    pauli_prod_op, pauli_prod_coef = pauli_product(pauli, pauli_other)
                    tot_coef = coef * coef_other * pauli_prod_coef
                    ret_op.add_term(pauli_prod_op, tot_coef)
            return ret_op
        else:
            return NotImplemented

    def __rmul__(self, other: object) -> "Operator":
        if not isinstance(other, (int, float, complex)):
            return NotImplemented
        return self * other

    def __truediv__(self, other: object) -> "Operator":
        if not isinstance(other, (int, float, complex)):
            return NotImplemented
        copied = self.copy()
        copied /= other
        return copied

    def __itruediv__(self, other: object) -> "Operator":
        if not isinstance(other, (int, float, complex)):
            return NotImplemented
        for pauli, coef in self.items():
            self[pauli] = coef / other
        return self

    def add_term(self, pauli_label: PauliLabel, coef: complex) -> None:
        """Method for adding single-term operator."""
        if coef == 0:
            return
        c = self.get(pauli_label, 0)
        new_coef = c + coef
        if new_coef == 0 and pauli_label in self:
            del self[pauli_label]
        else:
            self[pauli_label] = new_coef

    def copy(self) -> "Operator":
        """Returns the copy of itself."""
        return Operator(self)

    @property
    def n_terms(self) -> int:
        """Number of all terms."""
        return len(self)

    @property
    def constant(self) -> complex:
        """Constant value.

        Note that the constant value is a coefficient of an identity
        pauli term (PAULI_IDENTITY).
        """
        if PAULI_IDENTITY in self:
            return self[PAULI_IDENTITY]
        return 0.0

    @constant.setter
    def constant(self, new_val: complex) -> None:
        self[PAULI_IDENTITY] = new_val

    def hermitian_conjugated(self) -> "Operator":
        operator = Operator()
        for pauli, coef in self.items():
            operator[pauli] = coef.conjugate()
        return operator


def zero() -> Operator:
    """Returns zero (empty) operator, which always return zero expectation
    value for all states."""
    return Operator()


def commutator(op1: Operator, op2: Operator) -> Operator:
    """Returns the commutator of op1 and op2 :math:`[\\text{op1},

    \\text{op2}]`.
    """
    op_res = op1 * op2 - op2 * op1
    return op_res


def is_ops_close(
    op1: Operator, op2: Operator, rtol: float = 1e-9, atol: float = 0.0
) -> bool:
    """Returns ``True`` if two operators are close to each other."""
    uniq_p_labels = set(op1) | set(op2)

    for pauli in uniq_p_labels:
        op1_coeff = op1.get(pauli, 0.0)
        op2_coeff = op2.get(pauli, 0.0)
        if not cmath.isclose(op1_coeff, op2_coeff, rel_tol=rtol, abs_tol=atol):
            return False
    return True


def truncate(op: Operator, atol: float = 1e-8) -> Operator:
    """Returns truncated operator by eliminating terms whose coefficients are
    smaller than ``atol``."""
    _op = Operator()
    for pauli, coef in op.items():
        if abs(coef) >= atol:
            _op[pauli] = coef
    return _op


def is_hermitian(op: Operator, atol: float = 1e-8) -> bool:
    """Returns ``True`` if given operator is hermitian."""
    return is_ops_close(op, op.hermitian_conjugated(), atol)
