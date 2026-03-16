# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

import logging
import os
import random

import bpy
import gin

from infinigen.assets.humans.registry import register_human_actor
from infinigen.assets.static_assets.base import StaticAssetFactory
from infinigen.core.util.math import FixedSeed

logger = logging.getLogger(__name__)


def _warn_validation(msg: str, source_asset: str):
    logger.warning(f"[HumanAssetValidation] {msg} | source_asset={source_asset}")


def _iter_descendants(root_obj):
    stack = [root_obj]
    seen = set()
    while stack:
        obj = stack.pop()
        if obj in seen:
            continue
        seen.add(obj)
        yield obj
        stack.extend(obj.children)


class RiggedHumanFactory(StaticAssetFactory):
    """Factory for rigged human actors imported from FBX/GLB/BLEND files."""

    def __init__(
        self,
        factory_seed,
        asset_dir: str,
        coarse=False,
        body_shape_presets=(),
        clothing_variants=(),
        material_overrides=None,
        min_height_m: float = 1.2,
        max_height_m: float = 2.5,
    ):
        super().__init__(factory_seed, coarse)
        with FixedSeed(factory_seed):
            self.asset_dir = asset_dir
            self.body_shape_presets = list(body_shape_presets)
            self.clothing_variants = list(clothing_variants)
            self.material_overrides = material_overrides or {}
            self.min_height_m = min_height_m
            self.max_height_m = max_height_m

            self.asset_files = [
                f
                for f in os.listdir(self.asset_dir)
                if f.lower().endswith(("fbx", "glb", "gltf", "blend"))
            ]
            if len(self.asset_files) == 0:
                raise ValueError(f"No rigged character files found in {self.asset_dir}")

    def _import_asset(self, file_path: str) -> bpy.types.Object:
        extension = file_path.split(".")[-1].lower()
        func = self.import_map.get(extension)
        if func is None:
            raise ValueError(f"Unsupported file extension {extension} for {file_path}")

        initial_objects = set(bpy.context.scene.objects)
        if extension in ["glb", "gltf"]:
            func(filepath=file_path, merge_vertices=True)
        elif extension == "blend":
            self.import_single_object_from_blend(file_path)
        else:
            func(filepath=file_path)

        new_objects = set(bpy.context.scene.objects) - initial_objects
        if len(new_objects) == 0:
            raise ValueError(f"No object was imported from {file_path}")

        for obj in new_objects:
            obj.rotation_mode = "XYZ"

        armatures = [obj for obj in new_objects if obj.type == "ARMATURE"]
        if armatures:
            return armatures[0]

        roots = [obj for obj in new_objects if obj.parent is None]
        return roots[0] if roots else next(iter(new_objects))

    def _apply_body_shape_preset(self, root_obj: bpy.types.Object):
        if not self.body_shape_presets:
            return
        preset = random.choice(self.body_shape_presets)
        for obj in _iter_descendants(root_obj):
            if obj.type != "MESH":
                continue
            shape_keys = obj.data.shape_keys
            if shape_keys is None:
                continue
            for kb in shape_keys.key_blocks:
                kb.value = 1.0 if kb.name == preset else 0.0

    def _apply_clothing_variant(self, root_obj: bpy.types.Object):
        if not self.clothing_variants:
            return

        variant = random.choice(self.clothing_variants)
        tokens = [v.lower() for v in self.clothing_variants]
        for obj in _iter_descendants(root_obj):
            token_name = obj.name.lower()
            matched_any = any(token in token_name for token in tokens)
            if matched_any:
                obj.hide_render = variant.lower() not in token_name
                obj.hide_viewport = obj.hide_render

    def _apply_material_overrides(self, root_obj: bpy.types.Object):
        if not self.material_overrides:
            return

        for obj in _iter_descendants(root_obj):
            if obj.type != "MESH":
                continue
            for mat in obj.data.materials:
                if mat is None or mat.name not in self.material_overrides:
                    continue
                color = self.material_overrides[mat.name]
                if mat.node_tree is None:
                    continue
                principled = next(
                    (n for n in mat.node_tree.nodes if n.type == "BSDF_PRINCIPLED"),
                    None,
                )
                if principled is not None:
                    principled.inputs["Base Color"].default_value = color

    def _validate_asset(self, root_obj: bpy.types.Object, source_asset: str):
        has_armature = any(o.type == "ARMATURE" for o in _iter_descendants(root_obj))
        if not has_armature and root_obj.type != "ARMATURE":
            _warn_validation("Missing armature in imported actor", source_asset)

        dims = root_obj.dimensions
        max_dim = max(dims)
        if max_dim < self.min_height_m or max_dim > self.max_height_m:
            _warn_validation(
                (
                    "Unexpected actor scale"
                    f" (max_dim={max_dim:.3f}, expected [{self.min_height_m}, {self.max_height_m}])"
                ),
                source_asset,
            )

        rot = root_obj.rotation_euler
        if abs(rot[0]) > 0.5 or abs(rot[1]) > 0.5:
            _warn_validation(
                f"Unexpected orientation tilt (rotation_euler={tuple(rot)})", source_asset
            )

    def create_asset(self, **params) -> bpy.types.Object:
        asset_file = random.choice(self.asset_files)
        file_path = os.path.join(self.asset_dir, asset_file)
        imported_obj = self._import_asset(file_path)

        self._apply_body_shape_preset(imported_obj)
        self._apply_clothing_variant(imported_obj)
        self._apply_material_overrides(imported_obj)
        self._validate_asset(imported_obj, file_path)

        return imported_obj


@gin.configurable
def register_human_actor_asset_directory(
    category: str = "human_actor",
    asset_dir: str = "infinigen/assets/humans/source",
    weight: float = 1.0,
    body_shape_presets=(),
    clothing_variants=(),
    material_overrides=None,
    min_height_m: float = 1.2,
    max_height_m: float = 2.5,
):
    class ConfiguredRiggedHumanFactory(RiggedHumanFactory):
        def __init__(self, factory_seed, coarse=False):
            super().__init__(
                factory_seed=factory_seed,
                coarse=coarse,
                asset_dir=asset_dir,
                body_shape_presets=body_shape_presets,
                clothing_variants=clothing_variants,
                material_overrides=material_overrides,
                min_height_m=min_height_m,
                max_height_m=max_height_m,
            )

    ConfiguredRiggedHumanFactory.__name__ = f"{category.title().replace(' ', '')}Factory"
    register_human_actor(category, ConfiguredRiggedHumanFactory, weight=weight)
    return ConfiguredRiggedHumanFactory
