# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass
from enum import Enum, auto


class TimeUnit(Enum):
    SECOND = auto()
    MILLISECOND = auto()
    MICROSECOND = auto()
    NANOSECOND = auto()

    def __repr__(self) -> str:
        return "<%s.%s>" % (self.__class__.__name__, self.name)

    @staticmethod
    def from_str(name: str) -> "TimeUnit":
        return _time_name_table[name.lower()]

    def in_ns(self) -> float:
        return _ns_table[self]


_time_name_table = {
    "s": TimeUnit.SECOND,
    "ms": TimeUnit.MILLISECOND,
    "us": TimeUnit.MICROSECOND,
    "µs": TimeUnit.MICROSECOND,
    "ns": TimeUnit.NANOSECOND,
}

_ns_table = {
    TimeUnit.SECOND: 1e9,
    TimeUnit.MILLISECOND: 1e6,
    TimeUnit.MICROSECOND: 1e3,
    TimeUnit.NANOSECOND: 1,
}


@dataclass(frozen=True)
class TimeValue:
    value: float
    unit: TimeUnit

    def in_ns(self) -> float:
        return self.value * self.unit.in_ns()


class FrequencyUnit(Enum):
    HZ = auto()
    KHZ = auto()
    MHZ = auto()
    GHZ = auto()
    THZ = auto()

    def __repr__(self) -> str:
        return "<%s.%s>" % (self.__class__.__name__, self.name)

    @staticmethod
    def from_str(name: str) -> "FrequencyUnit":
        return _freq_name_table[name.lower()]

    def in_ghz(self) -> float:
        return _ghz_table[self]


_freq_name_table = {
    "hz": FrequencyUnit.HZ,
    "khz": FrequencyUnit.KHZ,
    "mhz": FrequencyUnit.MHZ,
    "ghz": FrequencyUnit.GHZ,
    "thz": FrequencyUnit.THZ,
}

_ghz_table = {
    FrequencyUnit.HZ: 1e-9,
    FrequencyUnit.KHZ: 1e-6,
    FrequencyUnit.MHZ: 1e-3,
    FrequencyUnit.GHZ: 1,
    FrequencyUnit.THZ: 1e3,
}


@dataclass(frozen=True)
class FrequencyValue:
    value: float
    unit: FrequencyUnit

    def in_ghz(self) -> float:
        return self.value * self.unit.in_ghz()
