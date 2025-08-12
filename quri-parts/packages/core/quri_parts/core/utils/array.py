# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import TYPE_CHECKING, Any, TypeVar, overload

import numpy as np

if TYPE_CHECKING:
    import numpy.typing as npt  # noqa: F401


_DType = TypeVar("_DType", bound=np.generic)


@overload
def readonly_array(array: "npt.ArrayLike") -> "npt.NDArray[Any]":
    ...


@overload
def readonly_array(array: "npt.NDArray[_DType]") -> "npt.NDArray[_DType]":
    ...


def readonly_array(array: "npt.ArrayLike") -> "npt.NDArray[Any]":
    """Returns a read-only view of the ``np.ndarray``."""
    view: "npt.NDArray[Any]" = np.asarray(array).view()
    view.flags.writeable = False
    return view
