# Licensed under the MIT License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      https://mit-license.org/
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from functools import wraps
from time import time
from typing import Callable, ParamSpec, Protocol, TypeVar


class HasElapsedTime(Protocol):
    elapsed_time: float | None


T = TypeVar("T", bound=HasElapsedTime)
P = ParamSpec("P")


def timer(f: Callable[P, T]) -> Callable[P, T]:
    @wraps(f)
    def wrap(*args: P.args, **kwargs: P.kwargs) -> T:
        t0 = time()
        result = f(*args, **kwargs)
        t1 = time()
        result.elapsed_time = t1 - t0
        return result

    return wrap
