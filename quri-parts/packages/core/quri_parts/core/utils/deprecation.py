# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import warnings
from functools import wraps
from typing import Any, Callable, TypeVar, cast

_F = TypeVar("_F", bound=Callable[..., Any])


def deprecated_shots_kwarg(func: _F) -> _F:
    """Accept the old ``n_shots`` keyword as a deprecated alias for ``shots``.

    Emits a :class:`DeprecationWarning` when ``n_shots`` is passed and
    forwards it to ``func`` as ``shots``.
    """

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        if "n_shots" in kwargs:
            warnings.warn(
                "The 'n_shots' keyword argument is deprecated and will be "
                "removed in a future release; use 'shots' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            kwargs["shots"] = kwargs.pop("n_shots")
        return func(*args, **kwargs)

    return cast(_F, wrapper)
