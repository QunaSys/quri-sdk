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


def deprecated_kwarg(old_name: str, new_name: str) -> Callable[[_F], _F]:
    """Build a decorator that accepts ``old_name`` as a deprecated alias for
    ``new_name``."""

    def decorator(func: _F) -> _F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if old_name in kwargs:
                if new_name in kwargs:
                    raise TypeError(
                        f"{func.__name__}() got both '{old_name}' and "
                        f"'{new_name}'; pass only '{new_name}'."
                    )
                warnings.warn(
                    f"The '{old_name}' keyword argument is deprecated and "
                    f"will be removed in a future release; use '{new_name}' "
                    "instead.",
                    DeprecationWarning,
                    stacklevel=2,
                )
                kwargs[new_name] = kwargs.pop(old_name)
            return func(*args, **kwargs)

        return cast(_F, wrapper)

    return decorator


#: Accept the old ``n_shots`` keyword as a deprecated alias for ``shots``.
deprecated_shots_kwarg = deprecated_kwarg("n_shots", "shots")

#: Accept the old ``measurement_cnt`` keyword as a deprecated alias for ``shots``.
deprecated_measurement_cnt_kwarg = deprecated_kwarg("measurement_cnt", "shots")
