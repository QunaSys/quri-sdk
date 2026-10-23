# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from time import sleep

from quri_parts.core.utils.concurrent import execute_concurrently


class TestExecuteConcurrently:
    def test_iterate_without_executor(self) -> None:
        r = execute_concurrently(lambda c, s: [c + i for i in s], 100, [1, 2, 3])
        assert r == [101, 102, 103]

    def test_iterate_with_thread_pool_executor(self) -> None:
        # Collect per-call chunks rather than keying on thread id: the pool may
        # reuse a worker for more than one chunk, which is not what is tested.
        chunks: list[tuple[int, ...]] = []

        def fn(c: int, s: Sequence[int]) -> Sequence[int]:
            x = [c + i for i in s]
            chunks.append(tuple(x))
            sleep(0.01)
            return x

        with ThreadPoolExecutor(max_workers=4) as executor:
            r = execute_concurrently(
                fn, 10, list(range(10)), executor=executor, concurrency=4
            )

        assert r == [10 + i for i in range(10)]
        assert sorted(chunks) == [(10, 11), (12, 13), (14, 15, 16), (17, 18, 19)]

    def test_iterate_with_process_pool_executor(self) -> None:
        with ProcessPoolExecutor(max_workers=4) as executor:
            r = execute_concurrently(
                _tag_with_chunk,
                10,
                list(range(10)),
                executor=executor,
                concurrency=4,
            )

        # Return value is: [((10, 11), 10), ((10, 11), 11), ((12, 13), 12), ...]
        assert [a for _, a in r] == [10 + i for i in range(10)]
        assert {chunk for chunk, _ in r} == {
            (10, 11),
            (12, 13),
            (14, 15, 16),
            (17, 18, 19),
        }


def _tag_with_chunk(c: int, s: Sequence[int]) -> Sequence[tuple[tuple[int, ...], int]]:
    # Worker processes share no memory with the test, so each result carries
    # the chunk it was computed in; grouping by pid would break when the pool
    # reuses a worker for a second chunk.
    chunk = tuple(c + i for i in s)
    return [(chunk, a) for a in chunk]
