# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pytest import approx

from quri_parts.core.utils.statistics import count_mean, count_var


class TestCountMean:
    def test_count_mean_int(self) -> None:
        mean = count_mean({1: 100, 2: 200, 4: 300})
        assert mean == approx(17.0 / 6)

    def test_count_mean_float(self) -> None:
        mean = count_mean({0.1: 100, 0.2: 200, 0.4: 300})
        assert mean == approx(1.7 / 6)

    def test_count_mean_complex(self) -> None:
        mean = count_mean({4 + 0.1j: 100, 2 + 0.2j: 200, 0.4j: 300})
        assert mean == approx((8 + 1.7j) / 6)


class TestCountVar:
    def test_count_var_int(self) -> None:
        var = count_var({1: 100, 2: 200, 4: 300})
        mean = 17.0 / 6
        expected = ((1 - mean) ** 2 + 2 * (2 - mean) ** 2 + 3 * (4 - mean) ** 2) / 6
        assert var == approx(expected)

    def test_count_var_float(self) -> None:
        var = count_var({0.1: 100, 0.2: 200, 0.4: 300})
        mean = 1.7 / 6
        expected = (
            (0.1 - mean) ** 2 + 2 * (0.2 - mean) ** 2 + 3 * (0.4 - mean) ** 2
        ) / 6
        assert var == approx(expected)

    def test_count_var_complex(self) -> None:
        var = count_var({4 + 0.1j: 100, 2 + 0.2j: 200, 0.4j: 300})
        mean_r = 8 / 6
        mean_i = 1.7 / 6
        expected = (
            ((4 - mean_r) ** 2 + (0.1 - mean_i) ** 2)
            + 2 * ((2 - mean_r) ** 2 + (0.2 - mean_i) ** 2)
            + 3 * ((0 - mean_r) ** 2 + (0.4 - mean_i) ** 2)
        ) / 6
        assert var == approx(expected)
