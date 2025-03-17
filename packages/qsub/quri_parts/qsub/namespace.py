from typing import NamedTuple, Optional


class NameSpace(NamedTuple):
    local_name: str
    parent: Optional["NameSpace"] = None

    @property
    def name(self) -> str:
        if self.parent:
            return ".".join((self.parent.name, self.local_name))
        else:
            return self.local_name

    def __str__(self) -> str:
        return self.name


#: A special "default" namespace used when a namespace is not specified
DEFAULT = NameSpace("__default__")
