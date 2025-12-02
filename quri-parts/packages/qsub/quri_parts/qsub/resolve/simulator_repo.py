# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from .resolve import CompositeSubRepository, SubRepository, default_repository


def _make_simulator_repo() -> CompositeSubRepository:
    from quri_parts.qsub.lib import std

    addition_resolver = (
        std.multi_control_gates.generate_multicontrolled_to_mc_sub_resolver()
    )
    addition_repo = SubRepository()
    addition_repo.register_sub_resolver(std.MultiControlled, addition_resolver)
    return CompositeSubRepository(default_repository(), addition_repo)


_SIMULATOR_REPO = _make_simulator_repo()


def simulator_repository() -> CompositeSubRepository:
    return _SIMULATOR_REPO
