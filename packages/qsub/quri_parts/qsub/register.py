from typing import NamedTuple


class Register(NamedTuple):
    uid: int

    def __str__(self) -> str:
        return f"r{self.uid}"
