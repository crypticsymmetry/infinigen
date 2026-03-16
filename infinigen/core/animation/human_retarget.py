# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

import json
import logging
from pathlib import Path

import bpy
import gin

logger = logging.getLogger(__name__)

DEFAULT_RETARGET_PROFILES = {
    "makehuman_basic": {
        "Hips": "hips",
        "Spine": "spine01",
        "Spine1": "spine02",
        "Neck": "neck",
        "Head": "head",
        "LeftShoulder": "clavicle_l",
        "LeftArm": "upperarm_l",
        "LeftForeArm": "lowerarm_l",
        "LeftHand": "hand_l",
        "RightShoulder": "clavicle_r",
        "RightArm": "upperarm_r",
        "RightForeArm": "lowerarm_r",
        "RightHand": "hand_r",
        "LeftUpLeg": "thigh_l",
        "LeftLeg": "calf_l",
        "LeftFoot": "foot_l",
        "RightUpLeg": "thigh_r",
        "RightLeg": "calf_r",
        "RightFoot": "foot_r",
    }
}


def _is_makehuman_rig(obj: bpy.types.Object) -> bool:
    if obj.type != "ARMATURE":
        return False
    name = obj.name.lower()
    return "makehuman" in name or "mhx" in name or bool(obj.get("makehuman"))


def _armature_scale(armature_obj: bpy.types.Object) -> float:
    lengths = [b.length for b in armature_obj.data.bones if b.use_deform]
    if not lengths:
        lengths = [b.length for b in armature_obj.data.bones]
    return sum(lengths) / max(1, len(lengths))


def _load_bvh_clip(clip_path: Path) -> bpy.types.Object:
    pre = {o.name for o in bpy.data.objects}
    bpy.ops.import_anim.bvh(filepath=str(clip_path), axis_forward="-Z", axis_up="Y")
    imported = [o for o in bpy.data.objects if o.name not in pre and o.type == "ARMATURE"]
    if not imported:
        raise RuntimeError(f"No armature imported from BVH clip {clip_path}")
    return imported[0]


def _apply_root_handling(rig: bpy.types.Object, root_bone: str, mode: str):
    if mode == "keep":
        return

    action = rig.animation_data.action if rig.animation_data else None
    if action is None:
        return

    datapath = f'pose.bones["{root_bone}"].location'
    fcurves = [
        action.fcurves.find(datapath, index=i)
        for i in range(3)
    ]
    if any(fc is None or len(fc.keyframe_points) == 0 for fc in fcurves):
        return

    start_xyz = [fc.keyframe_points[0].co[1] for fc in fcurves]

    if mode in {"in_place", "zero"}:
        offsets = start_xyz
    elif mode == "zero_xy":
        offsets = [start_xyz[0], start_xyz[1], 0.0]
    elif mode == "zero_z":
        offsets = [0.0, 0.0, start_xyz[2]]
    else:
        raise ValueError(f"Unsupported root handling mode: {mode}")

    for axis, fc in enumerate(fcurves):
        for kp in fc.keyframe_points:
            kp.co[1] -= offsets[axis]


def _retarget_clip_to_rig(
    source_armature: bpy.types.Object,
    target_rig: bpy.types.Object,
    mapping: dict[str, str],
    frame_start: int,
    frame_end: int,
    root_target_bone: str,
    include_root_motion: bool,
    root_handling: str,
):
    source_scale = _armature_scale(source_armature)
    target_scale = _armature_scale(target_rig)
    scale = target_scale / max(source_scale, 1e-8)
    source_armature.scale = (scale, scale, scale)
    bpy.context.view_layer.update()

    added_constraints = []
    for source_bone, target_bone in mapping.items():
        pb = target_rig.pose.bones.get(target_bone)
        if pb is None or source_bone not in source_armature.pose.bones:
            continue
        rot = pb.constraints.new("COPY_ROTATION")
        rot.target = source_armature
        rot.subtarget = source_bone
        added_constraints.append((pb, rot))

        if include_root_motion and target_bone == root_target_bone:
            loc = pb.constraints.new("COPY_LOCATION")
            loc.target = source_armature
            loc.subtarget = source_bone
            added_constraints.append((pb, loc))

    bpy.ops.object.select_all(action="DESELECT")
    target_rig.select_set(True)
    bpy.context.view_layer.objects.active = target_rig

    bpy.ops.nla.bake(
        frame_start=int(frame_start),
        frame_end=int(frame_end),
        only_selected=True,
        visual_keying=True,
        clear_constraints=True,
        use_current_action=True,
        bake_types={"POSE"},
    )

    # In case constraints were not auto-cleared by Blender for some versions.
    for pb, constraint in added_constraints:
        if constraint in pb.constraints:
            pb.constraints.remove(constraint)

    _apply_root_handling(target_rig, root_target_bone, root_handling)


@gin.configurable
def apply_makehuman_retargeting(
    output_folder: Path,
    clip_library_path: str = "",
    retarget_profile: str = "makehuman_basic",
    mapping_override: dict | None = None,
    actor_name_filter: str = "",
    frame_range: tuple[int, int] = (1, 120),
    clip_start_frame: int = 1,
    include_root_motion: bool = True,
    root_target_bone: str = "hips",
    root_handling: str = "in_place",
    metadata_filename: str = "animation_metadata.json",
):
    clip_root = Path(clip_library_path) if clip_library_path else None
    if clip_root is None or not clip_root.exists():
        logger.info("Retargeting skipped: clip_library_path does not exist")
        return []

    clips = sorted(clip_root.glob("*.bvh"))
    if not clips:
        logger.info("Retargeting skipped: no BVH clips found in %s", clip_root)
        return []

    profile_mapping = DEFAULT_RETARGET_PROFILES.get(retarget_profile, {})
    mapping = dict(profile_mapping)
    if mapping_override:
        mapping.update(mapping_override)

    rigs = [o for o in bpy.data.objects if _is_makehuman_rig(o)]
    if actor_name_filter:
        rigs = [o for o in rigs if actor_name_filter.lower() in o.name.lower()]

    if not rigs:
        logger.info("Retargeting skipped: no MakeHuman rigs in scene")
        return []

    actor_metadata = []
    clip_count = len(clips)

    for actor_idx, rig in enumerate(rigs):
        clip = clips[actor_idx % clip_count]
        source_armature = _load_bvh_clip(clip)

        clip_end = min(frame_range[1], clip_start_frame + int(source_armature.animation_data.action.frame_range[1]) - 1)
        _retarget_clip_to_rig(
            source_armature=source_armature,
            target_rig=rig,
            mapping=mapping,
            frame_start=max(frame_range[0], clip_start_frame),
            frame_end=clip_end,
            root_target_bone=root_target_bone,
            include_root_motion=include_root_motion,
            root_handling=root_handling,
        )

        actor_metadata.append(
            {
                "actor_name": rig.name,
                "clip_id": clip.stem,
                "clip_path": str(clip),
                "mapping_profile": retarget_profile,
                "frame_offset": int(max(frame_range[0], clip_start_frame) - frame_range[0]),
                "frame_range": [int(max(frame_range[0], clip_start_frame)), int(clip_end)],
                "root_handling": root_handling,
            }
        )

        bpy.data.objects.remove(source_armature, do_unlink=True)

    metadata_path = output_folder / metadata_filename
    metadata_path.write_text(json.dumps({"actors": actor_metadata}, indent=2))
    logger.info("Wrote retarget metadata to %s", metadata_path)
    return actor_metadata
