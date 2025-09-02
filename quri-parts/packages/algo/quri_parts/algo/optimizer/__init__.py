# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from .adam import AdaBelief, Adam
from .interface import (
    CostFunction,
    GradientFunction,
    Optimizer,
    OptimizerState,
    OptimizerStatus,
    Params,
)
from .lbfgs import LBFGS
from .nft import NFT, NFTfit
from .spsa import SPSA

#: Represents a parameter vector subject to optimization.
#: (A gradient vector is also represented as Params.)
Params = Params

#: Cost function for optimization.
CostFunction = CostFunction

#: Gradient function for optimization.
GradientFunction = GradientFunction


__all__ = [
    "Params",
    "CostFunction",
    "GradientFunction",
    "OptimizerStatus",
    "OptimizerState",
    "Optimizer",
    "Adam",
    "AdaBelief",
    "LBFGS",
    "NFT",
    "NFTfit",
    "SPSA",
]
