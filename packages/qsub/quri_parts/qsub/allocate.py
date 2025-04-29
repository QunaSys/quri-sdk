# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import Generic, TypeVar

from .qubit import Qubit
from .register import Register

T = TypeVar("T", Qubit, Register)


class Allocator(Generic[T], ABC):
    @abstractmethod
    def allocate(self, bit_count: int) -> Sequence[T]:
        ...

    @abstractmethod
    def allocate_map(self, bits: Sequence[T]) -> Mapping[T, T]:
        ...

    @abstractmethod
    def free(self, bits: Sequence[T]) -> None:
        ...

    @abstractmethod
    def free_last(self, bit_count: int) -> None:
        ...

    @abstractmethod
    def in_use(self, bit: T) -> bool:
        ...

    @abstractmethod
    def total(self) -> int:
        ...


class HierarchicalReuseAllocator(Allocator[T]):
    def __init__(self, bit_class: type[T], init_count: int = 0):
        self._bit_class: type[T] = bit_class
        self._index = init_count

    def allocate(self, bit_count: int) -> Sequence[T]:
        bits = [self._bit_class(i) for i in range(self._index, self._index + bit_count)]
        self._index += bit_count
        return bits

    def allocate_map(self, bits: Sequence[T]) -> Mapping[T, T]:
        dst = self.allocate(len(bits))
        return dict(zip(bits, dst))

    def free(self, bits: Sequence[T]) -> None:
        raise ValueError("This allocator cannot free arbitrary bits.")

    def free_last(self, bit_count: int) -> None:
        self._index -= bit_count

    def in_use(self, bit: T) -> bool:
        return bit.uid < self._index

    def total(self) -> int:
        return self._index


class QubitAllocator(HierarchicalReuseAllocator[Qubit]):
    def __init__(self, init_count: int = 0):
        super().__init__(Qubit, init_count)


class RegisterAllocator(HierarchicalReuseAllocator[Register]):
    def __init__(self, init_count: int = 0):
        super().__init__(Register, init_count)
