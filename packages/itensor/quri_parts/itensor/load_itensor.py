# This file must be in the same directory as the library.jl file
import os

_FILE = __file__
_ABS_DIR = os.path.dirname(os.path.abspath(_FILE))
_LIBRARY_PATH = os.path.join(_ABS_DIR, "library.jl")
_INCLUDE_STATEMENT = 'include("' + _LIBRARY_PATH + '")'


def ensure_itensor_loaded() -> None:
    """Ensure that the ITensor library is loaded into Juliacall."""
    from juliacall import Main as jl

    jl.seval(_INCLUDE_STATEMENT)
