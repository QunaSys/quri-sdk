# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import threading
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from os import getpid

from quri_parts.core.utils.concurrent import execute_concurrently


class TestExecuteConcurrently:
    def test_iterate_without_executor(self) -> None:
        r = execute_concurrently(lambda c, s: [c + i for i in s], 100, [1, 2, 3])
        assert r == [101, 102, 103]

    def test_iterate_with_thread_pool_executor(self) -> None:
        # Record (thread, chunk) per call rather than keying on thread id: the
        # pool may reuse a worker for more than one chunk.
        chunks: list[tuple[int, tuple[int, ...]]] = []

        def fn(c: int, s: Sequence[int]) -> Sequence[int]:
            x = [c + i for i in s]
            chunks.append((threading.get_ident(), tuple(x)))
            return x

        with ThreadPoolExecutor(max_workers=4) as executor:
            r = execute_concurrently(
                fn, 10, list(range(10)), executor=executor, concurrency=4
            )

        assert r == [10 + i for i in range(10)]
        assert sorted(chunk for _, chunk in chunks) == [
            (10, 11),
            (12, 13),
            (14, 15, 16),
            (17, 18, 19),
        ]
        # Every chunk must run on a worker thread, not inline on this one.
        assert all(t != threading.get_ident() for t, _ in chunks)

    def test_iterate_with_process_pool_executor(self) -> None:
        with ProcessPoolExecutor(max_workers=4) as executor:
            r = execute_concurrently(
                _tag_with_pid_and_chunk,
                10,
                list(range(10)),
                executor=executor,
                concurrency=4,
            )

        # Return value is: [(pid, (10, 11), 10), (pid, (10, 11), 11), ...]
        assert [a for _, _, a in r] == [10 + i for i in range(10)]
        assert {chunk for _, chunk, _ in r} == {
            (10, 11),
            (12, 13),
            (14, 15, 16),
            (17, 18, 19),
        }
        # Every chunk must run in a worker process, not inline in this one.
        assert all(p != getpid() for p, _, _ in r)


def _tag_with_pid_and_chunk(
    c: int, s: Sequence[int]
) -> Sequence[tuple[int, tuple[int, ...], int]]:
    # Worker processes share no memory with the test, so each result carries
    # the pid and chunk it was computed in. The chunk, not the pid, identifies
    # the call: the pool may reuse a worker for a second chunk.
    chunk = tuple(c + i for i in s)
    return [(getpid(), chunk, a) for a in chunk]
