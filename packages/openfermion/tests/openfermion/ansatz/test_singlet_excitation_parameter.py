# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from quri_parts.chem.utils.excitations import DoubleExcitation, SingleExcitation
from quri_parts.openfermion.ansatz.uccsd import _singlet_excitation_parameters


def single_excitation_checks(
    s_sz_symmetric_set: list[str],
    s_exc_param_fn_map: dict[SingleExcitation, list[tuple[str, float]]],
) -> None:
    for (i, a), param_fn in s_exc_param_fn_map.items():
        assert i % 2 == a % 2
        assert param_fn[0][0] in s_sz_symmetric_set
        assert param_fn[0][1] == 1
        assert param_fn[0][0] == f"s_{i//2}_{a//2}"


def double_excitation_checks(
    d_sz_symmetric_set: list[str],
    d_exc_param_fn_map: dict[DoubleExcitation, list[tuple[str, float]]],
) -> None:
    recovered_d_sz_symmetric_set = set()
    for (i, j, b, a), param_fn in d_exc_param_fn_map.items():
        if i % 2 == j % 2 == b % 2 == a % 2:
            # same spin excitation
            assert j > i and b > a
            assert len(param_fn) == 2
            assert (i // 2 != j // 2) and (a // 2 != b // 2)
            assert param_fn[0] == (f"d_{i//2}_{j//2}_{a//2}_{b//2}", 1)
            assert param_fn[1] == (f"d_{i//2}_{j//2}_{b//2}_{a//2}", -1)
            recovered_d_sz_symmetric_set.add(f"d_{i//2}_{j//2}_{a//2}_{b//2}")
            recovered_d_sz_symmetric_set.add(f"d_{i//2}_{j//2}_{b//2}_{a//2}")

        elif (i % 2 + 1 == j % 2) and (b % 2 == a % 2 + 1):
            # mixed spin excitation
            assert len(param_fn) == 1
            assert d_exc_param_fn_map[(j - 1, i + 1, a + 1, b - 1)] == param_fn
            assert param_fn[0][0] in d_sz_symmetric_set
            assert param_fn[0][1] == 1
            recovered_d_sz_symmetric_set.add(param_fn[0][0])
        else:
            assert False

    assert recovered_d_sz_symmetric_set == set(d_sz_symmetric_set)


def test_parameter_4_2() -> None:
    """Test no same spin excitation."""
    n_occ = 1
    n_vir = 1

    (
        s_sz_symmetric_set,
        s_exc_param_fn_map,
        d_sz_symmetric_set,
        d_exc_param_fn_map,
    ) = _singlet_excitation_parameters(2 * (n_occ + n_vir), n_occ * 2)

    assert len(s_sz_symmetric_set) == n_occ * n_vir
    assert len(d_sz_symmetric_set) == (n_occ * n_vir) * (n_occ * n_vir + 1) / 2
    single_excitation_checks(s_sz_symmetric_set, s_exc_param_fn_map)
    double_excitation_checks(d_sz_symmetric_set, d_exc_param_fn_map)


def test_parameter_8_4() -> None:
    """Test 1 pair of same spin excitation."""

    n_occ = 2
    n_vir = 2

    (
        s_sz_symmetric_set,
        s_exc_param_fn_map,
        d_sz_symmetric_set,
        d_exc_param_fn_map,
    ) = _singlet_excitation_parameters(2 * (n_occ + n_vir), n_occ * 2)

    assert len(s_sz_symmetric_set) == n_occ * n_vir
    assert len(d_sz_symmetric_set) == (n_occ * n_vir) * (n_occ * n_vir + 1) / 2
    single_excitation_checks(s_sz_symmetric_set, s_exc_param_fn_map)
    double_excitation_checks(d_sz_symmetric_set, d_exc_param_fn_map)


def test_parameter_12_6() -> None:
    """Test multiple pairs of same spin excitation."""

    n_occ = 3
    n_vir = 3

    (
        s_sz_symmetric_set,
        s_exc_param_fn_map,
        d_sz_symmetric_set,
        d_exc_param_fn_map,
    ) = _singlet_excitation_parameters(2 * (n_occ + n_vir), n_occ * 2)

    assert len(s_sz_symmetric_set) == n_occ * n_vir
    assert len(d_sz_symmetric_set) == (n_occ * n_vir) * (n_occ * n_vir + 1) / 2
    single_excitation_checks(s_sz_symmetric_set, s_exc_param_fn_map)
    double_excitation_checks(d_sz_symmetric_set, d_exc_param_fn_map)
