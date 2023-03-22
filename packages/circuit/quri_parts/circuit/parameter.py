# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


class Parameter:
    """A class representing parameters in parametric quantum circuits.

    A ``Parameter`` is a placeholder and does not hold a concrete value in it.

    Implementation note: equality of Parameters is evaluated as identity of the objects.
    This means that even if two Parameters have the same name they are not equal if
    they are different objects. To achieve this behavior, it is avoided to 1) inherit
    :class:`~NamedTuple` and 2) define ``__eq__`` method.
    """

    def __init__(self, name: str = ""):
        self.name = name

    def __repr__(self) -> str:
        return f"Parameter(name={self.name})"


#: A placeholder representing a constant term.
CONST = Parameter()
