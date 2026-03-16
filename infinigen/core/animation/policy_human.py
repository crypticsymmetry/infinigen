# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
from shapely.geometry import Point
import trimesh

from infinigen.core import tags as t

logger = logging.getLogger(__name__)


ROOM_TAGS = {
    t.Semantics.Kitchen,
    t.Semantics.Bedroom,
    t.Semantics.LivingRoom,
    t.Semantics.Closet,
    t.Semantics.Hallway,
    t.Semantics.Bathroom,
    t.Semantics.Garage,
    t.Semantics.Balcony,
    t.Semantics.DiningRoom,
    t.Semantics.Utility,
    t.Semantics.StaircaseRoom,
    t.Semantics.Warehouse,
    t.Semantics.Office,
    t.Semantics.MeetingRoom,
    t.Semantics.OpenOffice,
    t.Semantics.BreakRoom,
    t.Semantics.Restroom,
    t.Semantics.FactoryOffice,
}

SUPPORT_LIKE_TAGS = {
    t.Semantics.Table,
    t.Semantics.Desk,
    t.Semantics.SideTable,
    t.Semantics.KitchenCounter,
    t.Semantics.Storage,
    t.Semantics.Bed,
}


@dataclass(slots=True)
class MocapClipCandidate:
    """Describes a mocap clip and optional environment constraints/preferences."""

    clip_id: str
    keyframes: list[tuple[float, float, float]] = field(default_factory=list)
    room_types: set[t.Semantics | str] = field(default_factory=set)
    support_tags: set[t.Tag] = field(default_factory=set)
    prop_tags: set[t.Tag] = field(default_factory=set)
    min_free_floor_area: float = 0.0
    keyframes_world_space: bool = False
    metadata: dict = field(default_factory=dict)


@dataclass(slots=True)
class PolicyDecision:
    clip_id: str
    confidence: float
    fallback_used: bool
    fallback_reason: str | None
    score_breakdown: dict[str, float]


@dataclass(slots=True)
class SceneContext:
    room_type: t.Semantics | None
    free_floor_area: float | None
    nearest_support_dist: float | None
    nearby_prop_tags: set[t.Tag]


class HumanAnimationPolicy:
    """Selects mocap clips using indoor-solver semantics and collision validation."""

    def __init__(
        self,
        support_search_radius: float = 2.5,
        prop_search_radius: float = 2.5,
        collision_clearance: float = 0.08,
        floor_occupancy_ratio: float = 0.9,
        rng_seed: int | None = None,
    ):
        self.support_search_radius = support_search_radius
        self.prop_search_radius = prop_search_radius
        self.collision_clearance = collision_clearance
        self.floor_occupancy_ratio = floor_occupancy_ratio
        self.rng = np.random.default_rng(rng_seed)

    def select_clip(
        self,
        state,
        actor_name: str,
        candidates: Sequence[MocapClipCandidate],
        scene_metadata_path: str | Path | None = None,
    ) -> PolicyDecision:
        if not candidates:
            raise ValueError("select_clip called with no mocap candidates")

        actor_state = state.objs[actor_name]
        context = self._build_scene_context(state, actor_name)

        scored = []
        for candidate in candidates:
            score, breakdown = self._score_candidate(candidate, context)
            scored.append((score, candidate, breakdown))
        scored.sort(key=lambda x: x[0], reverse=True)

        best_score, _, _ = scored[0]
        fallback_reason = self._fallback_reason(context, best_score)
        fallback_used = fallback_reason is not None

        ranked_candidates = [item[1] for item in scored]
        ranked_breakdowns = {item[1].clip_id: item[2] for item in scored}
        valid_candidates = [
            c for c in ranked_candidates if self._validate_keyframes(c, actor_state, state)
        ]

        if not valid_candidates:
            selected_candidate = self.rng.choice(list(candidates))
            fallback_used = True
            fallback_reason = fallback_reason or "all_candidates_failed_collision_validation"
        elif fallback_used:
            selected_candidate = self.rng.choice(valid_candidates)
        else:
            selected_candidate = valid_candidates[0]

        confidence = self._compute_confidence(scored, selected_candidate.clip_id)

        decision = PolicyDecision(
            clip_id=selected_candidate.clip_id,
            confidence=confidence,
            fallback_used=fallback_used,
            fallback_reason=fallback_reason,
            score_breakdown=ranked_breakdowns[selected_candidate.clip_id],
        )

        if scene_metadata_path is not None:
            self._write_policy_metadata(scene_metadata_path, actor_name, decision, context)

        return decision

    def _build_scene_context(self, state, actor_name: str) -> SceneContext:
        actor_state = state.objs[actor_name]
        actor_xy = self._xy_from_object_state(actor_state)
        room_obj = self._find_room_for_actor(state, actor_name, actor_xy)

        room_type = None
        free_floor_area = None
        if room_obj is not None:
            room_type = next((tg for tg in room_obj.tags if tg in ROOM_TAGS), None)
            free_floor_area = self._estimate_free_floor_area(state, room_obj)

        nearest_support_dist = self._nearest_object_distance(
            state,
            actor_xy,
            required_tags={t.Semantics.Object},
            any_tags=SUPPORT_LIKE_TAGS | {t.Subpart.SupportSurface},
            max_dist=self.support_search_radius,
        )

        nearby_prop_tags = self._collect_nearby_prop_tags(state, actor_xy)

        return SceneContext(
            room_type=room_type,
            free_floor_area=free_floor_area,
            nearest_support_dist=nearest_support_dist,
            nearby_prop_tags=nearby_prop_tags,
        )

    def _score_candidate(
        self, candidate: MocapClipCandidate, context: SceneContext
    ) -> tuple[float, dict[str, float]]:
        breakdown = {
            "room_semantics": 0.0,
            "support_affordance": 0.0,
            "free_floor_area": 0.0,
            "nearby_props": 0.0,
        }

        candidate_room_types = {
            t.Semantics(rt)
            if isinstance(rt, str) and rt in t.Semantics._value2member_map_
            else rt
            for rt in candidate.room_types
        }

        if context.room_type is not None and candidate_room_types:
            breakdown["room_semantics"] = 1.0 if context.room_type in candidate_room_types else -0.3

        if candidate.support_tags:
            if context.nearest_support_dist is None:
                breakdown["support_affordance"] = -0.5
            else:
                d = context.nearest_support_dist
                breakdown["support_affordance"] = float(
                    np.clip(1.0 - d / max(self.support_search_radius, 1e-3), -0.5, 1.0)
                )

        if candidate.min_free_floor_area > 0 and context.free_floor_area is not None:
            ratio = context.free_floor_area / max(candidate.min_free_floor_area, 1e-6)
            breakdown["free_floor_area"] = float(np.clip(ratio - 1.0, -1.0, 1.0))

        if candidate.prop_tags:
            overlap = len(set(candidate.prop_tags).intersection(context.nearby_prop_tags))
            breakdown["nearby_props"] = overlap / max(len(candidate.prop_tags), 1)

        weights = {
            "room_semantics": 0.4,
            "support_affordance": 0.25,
            "free_floor_area": 0.2,
            "nearby_props": 0.15,
        }
        score = float(sum(breakdown[k] * w for k, w in weights.items()))
        return score, breakdown

    def _fallback_reason(self, context: SceneContext, best_score: float) -> str | None:
        missing = []
        if context.room_type is None:
            missing.append("missing_room_semantics")
        if context.free_floor_area is None:
            missing.append("missing_floor_area")
        if context.nearest_support_dist is None:
            missing.append("missing_support_surface")
        if not context.nearby_prop_tags:
            missing.append("missing_nearby_props")

        if len(missing) >= 3:
            return ",".join(missing)
        if best_score < 0.05:
            return "insufficient_semantic_signal"
        return None

    def _validate_keyframes(self, candidate: MocapClipCandidate, actor_state, state) -> bool:
        if not candidate.keyframes:
            return True

        combined_mesh = self._combined_scene_mesh(state)
        if combined_mesh is None:
            logger.warning(
                "policy_human: collision validation skipped because no trimesh geometry is available"
            )
            return True

        origin = np.array(actor_state.obj.location, dtype=float)
        for keyframe in candidate.keyframes:
            key = np.array(keyframe, dtype=float)
            key_world = key if candidate.keyframes_world_space else origin + key
            try:
                signed_dist = trimesh.proximity.signed_distance(
                    combined_mesh, key_world.reshape(1, 3)
                )[0]
            except BaseException:
                logger.exception("policy_human: signed distance failed, rejecting clip")
                return False

            if signed_dist < self.collision_clearance:
                return False
        return True

    @staticmethod
    def _combined_scene_mesh(state):
        scene = getattr(state, "trimesh_scene", None)
        if scene is None:
            return None

        meshes = [
            geom
            for name, geom in scene.geometry.items()
            if isinstance(geom, trimesh.Trimesh) and not name.startswith("camera")
        ]
        if not meshes:
            return None

        try:
            return trimesh.util.concatenate(meshes)
        except BaseException:
            logger.exception("policy_human: failed to merge meshes for collision checks")
            return None

    def _find_room_for_actor(self, state, actor_name: str, actor_xy: np.ndarray):
        actor_state = state.objs[actor_name]
        for rel in actor_state.relations:
            if rel.target_name in state.objs:
                target = state.objs[rel.target_name]
                if t.Semantics.Room in target.tags:
                    return target

        candidate_rooms = [
            os
            for os in state.objs.values()
            if t.Semantics.Room in os.tags and os.polygon is not None
        ]
        if not candidate_rooms:
            return None

        point = Point(*actor_xy)
        containing_rooms = [r for r in candidate_rooms if r.polygon.buffer(1e-3).contains(point)]
        if containing_rooms:
            return containing_rooms[0]

        return min(candidate_rooms, key=lambda r: r.polygon.distance(point))

    def _estimate_free_floor_area(self, state, room_state) -> float | None:
        if room_state.polygon is None:
            return None

        room_area = float(room_state.polygon.area)
        occupied = 0.0
        for os in state.objs.values():
            if os is room_state or os.polygon is None:
                continue
            if t.Semantics.Object not in os.tags:
                continue
            occupied += os.polygon.intersection(room_state.polygon).area

        return max(room_area - occupied * self.floor_occupancy_ratio, 0.0)

    def _nearest_object_distance(
        self,
        state,
        origin_xy: np.ndarray,
        required_tags: set[t.Tag],
        any_tags: set[t.Tag] | None,
        max_dist: float,
    ) -> float | None:
        min_dist = None
        origin_point = Point(*origin_xy)
        for os in state.objs.values():
            if os.polygon is None:
                continue
            if not required_tags.issubset(os.tags):
                continue
            if any_tags is not None and not any_tags.intersection(os.tags):
                continue
            dist = os.polygon.distance(origin_point)
            if dist > max_dist:
                continue
            min_dist = dist if min_dist is None else min(min_dist, dist)
        return float(min_dist) if min_dist is not None else None

    def _collect_nearby_prop_tags(self, state, origin_xy: np.ndarray) -> set[t.Tag]:
        nearby_tags: set[t.Tag] = set()
        origin_point = Point(*origin_xy)
        for os in state.objs.values():
            if os.polygon is None:
                continue
            if os.polygon.distance(origin_point) > self.prop_search_radius:
                continue
            nearby_tags.update(
                tag
                for tag in os.tags
                if isinstance(tag, t.Semantics)
                and tag not in ROOM_TAGS
                and tag not in {t.Semantics.Object, t.Semantics.Room}
            )
        return nearby_tags

    @staticmethod
    def _xy_from_object_state(object_state) -> np.ndarray:
        return np.array(object_state.obj.location[:2], dtype=float)

    @staticmethod
    def _compute_confidence(
        scored: Sequence[tuple[float, MocapClipCandidate, dict]], selected_clip_id: str
    ) -> float:
        scores = np.array([s for s, _, _ in scored], dtype=float)
        if scores.size == 0:
            return 0.0

        top = float(np.max(scores))
        second = float(np.partition(scores, -2)[-2]) if len(scores) > 1 else -1.0
        margin = top - second
        confidence = float(np.clip(0.5 + 0.4 * top + 0.1 * margin, 0.0, 1.0))

        if selected_clip_id != scored[0][1].clip_id:
            confidence *= 0.8

        return confidence

    def _write_policy_metadata(
        self,
        metadata_path: str | Path,
        actor_name: str,
        decision: PolicyDecision,
        context: SceneContext,
    ):
        metadata_path = Path(metadata_path)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)

        if metadata_path.exists():
            with metadata_path.open("r") as f:
                metadata = json.load(f)
        else:
            metadata = {}

        metadata.setdefault("human_animation_policy", {})[actor_name] = {
            "decision": asdict(decision),
            "context": {
                "room_type": str(context.room_type) if context.room_type is not None else None,
                "free_floor_area": context.free_floor_area,
                "nearest_support_dist": context.nearest_support_dist,
                "nearby_prop_tags": sorted(str(tag) for tag in context.nearby_prop_tags),
            },
        }

        with metadata_path.open("w") as f:
            json.dump(metadata, f, indent=2, sort_keys=True)


def candidate_from_dict(data: dict) -> MocapClipCandidate:
    """Utility for loading candidates from JSON-like dictionaries."""

    return MocapClipCandidate(
        clip_id=data["clip_id"],
        keyframes=[tuple(v) for v in data.get("keyframes", [])],
        room_types=set(data.get("room_types", [])),
        support_tags={t.Semantics(v) for v in data.get("support_tags", [])},
        prop_tags={t.Semantics(v) for v in data.get("prop_tags", [])},
        min_free_floor_area=float(data.get("min_free_floor_area", 0.0)),
        keyframes_world_space=bool(data.get("keyframes_world_space", False)),
        metadata=dict(data.get("metadata", {})),
    )
