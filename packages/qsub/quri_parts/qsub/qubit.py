from typing import NamedTuple


class Qubit(NamedTuple):
    uid: int

    def __str__(self) -> str:
        return f"q{self.uid}"
