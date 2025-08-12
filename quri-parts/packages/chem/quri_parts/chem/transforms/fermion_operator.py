# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sympy

from quri_parts.chem.transforms.fermion import FermionLabel

COEFFICIENT_TYPES = (int, float, complex, sympy.Expr)


class FermionOperator(dict[FermionLabel, complex]):
    def __str__(self) -> str:
        return " + ".join(f"{coef} [{label}]" for label, coef in self.items())

    def add_term(self, fermion_label: FermionLabel, coef: complex) -> None:
        """Method for adding single-term operator."""
        if coef == 0:
            return
        c = self.get(fermion_label, 0)
        new_coef = c + coef
        if new_coef == 0 and fermion_label in self:
            del self[fermion_label]
        else:
            self[fermion_label] = new_coef

    def copy(self) -> "FermionOperator":
        """Returns the copy of itself."""
        return FermionOperator(self)

    @property
    def n_terms(self) -> int:
        """Number of all terms."""
        return len(self)

    @property
    def constant(self) -> complex:
        """The value of the constant term."""
        return self.get(FermionLabel(), complex(0.0))

    @constant.setter
    def constant(self, value):  # type: ignore
        self[FermionLabel()] = value

    def __mul__(self, multiplier: object) -> "FermionOperator":
        if isinstance(multiplier, COEFFICIENT_TYPES):
            return FermionOperator(
                {fermion: multiplier * coef for fermion, coef in self.items()}
            )
        else:
            return NotImplemented

    def __rmul__(self, multiplier: object) -> "FermionOperator":
        if not isinstance(multiplier, COEFFICIENT_TYPES):
            return NotImplemented
        return self * multiplier

    def __sub__(self, subtrahend: object) -> "FermionOperator":
        if not isinstance(subtrahend, COEFFICIENT_TYPES):
            return NotImplemented
        copied = self.copy()
        copied -= subtrahend
        return copied

    def __add__(self, addend: object) -> "FermionOperator":
        if not isinstance(addend, COEFFICIENT_TYPES):
            return NotImplemented
        copied = self.copy()
        copied += addend
        return copied

    def __isub__(self, subtrahend: object) -> "FermionOperator":
        if not isinstance(subtrahend, COEFFICIENT_TYPES):
            return NotImplemented
        self.constant -= subtrahend
        return self

    def __iadd__(self, addend: object) -> "FermionOperator":
        if isinstance(addend, COEFFICIENT_TYPES):
            self.constant += addend
        elif isinstance(addend, FermionOperator):
            for fermion, coef in addend.items():
                self.add_term(fermion, coef)
        else:
            return NotImplemented
        return self

    def __rsub__(self, subtrahend: object) -> "FermionOperator":
        return -1 * self + subtrahend

    def __radd__(self, addend: object) -> "FermionOperator":
        return self + addend
