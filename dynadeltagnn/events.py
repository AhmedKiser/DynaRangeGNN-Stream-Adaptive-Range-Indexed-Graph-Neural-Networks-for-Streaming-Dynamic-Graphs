from __future__ import annotations

from dataclasses import dataclass
from torch import Tensor


@dataclass(frozen=True)
class FeatureUpdate:
    node: int
    new_value: Tensor


@dataclass(frozen=True)
class EdgeAdd:
    u: int
    v: int


@dataclass(frozen=True)
class EdgeDelete:
    u: int
    v: int
