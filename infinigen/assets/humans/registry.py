# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

from dataclasses import dataclass


@dataclass(frozen=True)
class HumanActorRegistration:
    category: str
    factory_cls: type
    weight: float = 1.0


_REGISTRY: dict[str, HumanActorRegistration] = {}


def register_human_actor(category: str, factory_cls: type, weight: float = 1.0):
    if weight <= 0:
        raise ValueError(f"Expected positive weight for {category}, got {weight}")
    _REGISTRY[category] = HumanActorRegistration(
        category=category, factory_cls=factory_cls, weight=weight
    )


def get_registered_human_actors() -> dict[str, HumanActorRegistration]:
    return dict(_REGISTRY)
