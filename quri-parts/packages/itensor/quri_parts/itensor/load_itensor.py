# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This file must be in the same directory as the library.jl file
import os

_FILE = __file__
_ABS_DIR = os.path.dirname(os.path.abspath(_FILE))
_LIBRARY_PATH = os.path.join(_ABS_DIR, "library.jl")
_INCLUDE_STATEMENT = 'include("' + _LIBRARY_PATH + '")'

_is_jl_library_included = False


def ensure_itensor_loaded() -> None:
    """Ensure that the ITensor library is loaded into Juliacall."""
    global _is_jl_library_included

    if not _is_jl_library_included:
        from juliacall import Main as jl

        jl.seval(_INCLUDE_STATEMENT)
        _is_jl_library_included = True
