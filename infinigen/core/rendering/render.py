# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - Lahav Lipson - Render, flat shading, etc
# - Alex Raistrick - Compositing
# - Hei Law - Initial version


import json
import logging
import os
import time
from fnmatch import fnmatch
from pathlib import Path
from typing import Literal

import bpy
import gin
import numpy as np
from imageio import imwrite
import imageio.v3 as iio

from infinigen.core import init
from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler
from infinigen.core.placement import camera as cam_util
from infinigen.core.rendering.post_render import (
    colorize_depth,
    colorize_flow,
    colorize_int_array,
    colorize_normals,
    load_depth,
    load_flow,
    load_normals,
    load_seg_mask,
    load_uniq_inst,
)
from infinigen.core.util.blender import set_geometry_option
from infinigen.core.util.logging import Timer
from infinigen.tools.datarelease_toolkit import reorganize_old_framesfolder
from infinigen.tools.suffixes import get_suffix

TRANSPARENT_SHADERS = {Nodes.TranslucentBSDF, Nodes.TransparentBSDF}

logger = logging.getLogger(__name__)


def remove_translucency():
    # The asserts were added since these edge cases haven't appeared yet -Lahav
    for material in bpy.data.materials:
        nw = NodeWrangler(material.node_tree)
        for node in nw.nodes:
            if node.bl_idname == Nodes.MixShader:
                fac_soc, shader_1_soc, shader_2_soc = node.inputs
                assert shader_1_soc.is_linked and len(shader_1_soc.links) == 1
                assert shader_2_soc.is_linked and len(shader_2_soc.links) == 1
                shader_1_type = shader_1_soc.links[0].from_node.bl_idname
                shader_2_type = shader_2_soc.links[0].from_node.bl_idname
                assert not (
                    shader_1_type in TRANSPARENT_SHADERS
                    and shader_2_type in TRANSPARENT_SHADERS
                )
                if shader_1_type in TRANSPARENT_SHADERS:
                    assert not fac_soc.is_linked
                    fac_soc.default_value = 1.0
                elif shader_2_type in TRANSPARENT_SHADERS:
                    assert not fac_soc.is_linked
                    fac_soc.default_value = 0.0


def set_pass_indices():
    tree_output = {}
    index = 1
    for obj in bpy.data.objects:
        if obj.hide_render:
            continue
        if obj.pass_index == 0:
            obj.pass_index = index
            index += 1
        object_dict = {"type": obj.type, "object_index": obj.pass_index, "children": []}
        if obj.type == "MESH":
            object_dict["num_verts"] = len(obj.data.vertices)
            object_dict["num_faces"] = len(obj.data.polygons)
            object_dict["materials"] = obj.material_slots.keys()
            object_dict["unapplied_modifiers"] = obj.modifiers.keys()
        tree_output[obj.name] = object_dict
        for child_obj in obj.children:
            if child_obj.pass_index == 0:
                child_obj.pass_index = index
                index += 1
            object_dict["children"].append(child_obj.pass_index)
        index += 1
    return tree_output


def set_material_pass_indices():
    output_material_properties = {}
    mat_index = 1
    for mat in bpy.data.materials:
        if mat.pass_index == 0:
            mat.pass_index = mat_index
            mat_index += 1
        output_material_properties[mat.name] = {"pass_index": mat.pass_index}
    return output_material_properties


# Can be pasted directly into the blender console
def make_clay():
    clay_material = bpy.data.materials.new(name="clay")
    clay_material.diffuse_color = (0.2, 0.05, 0.01, 1)
    for obj in bpy.data.objects:
        if "atmosphere" not in obj.name.lower() and not obj.hide_render:
            if len(obj.material_slots) == 0:
                obj.active_material = clay_material
            else:
                for mat_slot in obj.material_slots:
                    mat_slot.material = clay_material


@gin.configurable
def compositor_postprocessing(
    nw,
    source,
    show=True,
    color_correct=True,
    distort=0,
    glare=False,
):
    if distort > 0:
        source = nw.new_node(
            Nodes.LensDistortion, input_kwargs={"Image": source, "Dispersion": distort}
        )

    if color_correct:
        source = nw.new_node(
            Nodes.BrightContrast,
            input_kwargs={"Image": source, "Bright": 1.0, "Contrast": 4.0},
        )

    if glare:
        source = nw.new_node(
            Nodes.Glare,
            input_kwargs={"Image": source},
            attrs={"glare_type": "GHOSTS", "threshold": 0.5, "mix": -0.99},
        )

    if show:
        nw.new_node(Nodes.Composite, input_kwargs={"Image": source})

    return source.outputs[0] if hasattr(source, "outputs") else source


@gin.configurable
def configure_compositor_output(
    nw,
    frames_folder,
    image_denoised,
    image_noisy,
    passes_to_save,
    saving_ground_truth,
    output_prefix="",
    save_alpha=False,
    binary_mask_threshold=0.5,
):
    file_output_node_png = nw.new_node(
        Nodes.OutputFile,
        attrs={
            "base_path": str(frames_folder),
            "format.file_format": "PNG",
            "format.color_mode": "RGB",
        },
    )
    file_output_node_exr = nw.new_node(
        Nodes.OutputFile,
        attrs={
            "base_path": str(frames_folder),
            "format.file_format": "OPEN_EXR",
            "format.color_mode": "RGB",
        },
    )
    default_file_output_node = (
        file_output_node_exr if saving_ground_truth else file_output_node_png
    )
    file_slot_list = []
    viewlayer = bpy.context.scene.view_layers["ViewLayer"]
    render_layers = nw.new_node(Nodes.RenderLayers)
    for viewlayer_pass, socket_name in passes_to_save:
        if hasattr(viewlayer, f"use_pass_{viewlayer_pass}"):
            setattr(viewlayer, f"use_pass_{viewlayer_pass}", True)
        else:
            setattr(viewlayer.cycles, f"use_pass_{viewlayer_pass}", True)
        # must save the material pass index as EXR
        file_output_node = (
            default_file_output_node
            if viewlayer_pass != "material_index"
            else file_output_node_exr
        )

        slot_input = file_output_node.file_slots.new(f"{output_prefix}{socket_name}")
        render_socket = render_layers.outputs[socket_name]
        match viewlayer_pass:
            case "vector":
                separate_color = nw.new_node(Nodes.CompSeparateColor, [render_socket])
                comnbine_color = nw.new_node(
                    Nodes.CompCombineColor,
                    [0, (separate_color, 3), (separate_color, 2), 0],
                )
                nw.links.new(comnbine_color.outputs[0], slot_input)
            case "normal":
                color = nw.new_node(
                    Nodes.CompositorMixRGB,
                    [None, render_socket, (0, 0, 0, 0)],
                    attrs={"blend_type": "ADD"},
                ).outputs[0]
                nw.links.new(color, slot_input)
            case _:
                nw.links.new(render_socket, slot_input)
        file_slot_list.append(file_output_node.file_slots[slot_input.name])

    slot_name = "Image"
    slot_input = default_file_output_node.file_slots[slot_name]
    slot_input.path = f"{output_prefix}Image"
    image = image_denoised if image_denoised is not None else image_noisy
    nw.links.new(image, default_file_output_node.inputs["Image"])
    if saving_ground_truth:
        slot_input.path = f"{output_prefix}UniqueInstances"
    else:
        nw.links.new(image, file_output_node_exr.inputs["Image"])
        file_output_node_exr.file_slots[slot_name].path = f"{output_prefix}Image"
        file_slot_list.append(file_output_node_exr.file_slots[slot_name])
    file_slot_list.append(default_file_output_node.file_slots[slot_name])

    if save_alpha:
        alpha_output_node = nw.new_node(
            Nodes.OutputFile,
            attrs={
                "base_path": str(frames_folder),
                "format.file_format": "PNG",
                "format.color_mode": "BW",
            },
        )
        alpha_slot = alpha_output_node.file_slots.new(f"{output_prefix}AlphaMatte")
        alpha_slot.path = f"{output_prefix}AlphaMatte"
        nw.links.new(render_layers.outputs["Alpha"], alpha_slot)
        file_slot_list.append(alpha_output_node.file_slots[alpha_slot.name])

        binary_output_node = nw.new_node(
            Nodes.OutputFile,
            attrs={
                "base_path": str(frames_folder),
                "format.file_format": "PNG",
                "format.color_mode": "BW",
            },
        )
        binary_slot = binary_output_node.file_slots.new(f"{output_prefix}BinaryMask")
        binary_slot.path = f"{output_prefix}BinaryMask"
        binary_mask = nw.new_node(
            Nodes.Math,
            [render_layers.outputs["Alpha"], binary_mask_threshold],
            attrs={"operation": "GREATER_THAN"},
        )
        nw.links.new(binary_mask.outputs[0], binary_slot)
        file_slot_list.append(binary_output_node.file_slots[binary_slot.name])

    return file_slot_list


def shader_random(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    object_info = nw.new_node(Nodes.ObjectInfo_Shader)

    white_noise_texture = nw.new_node(
        Nodes.WhiteNoiseTexture, input_kwargs={"Vector": object_info.outputs["Random"]}
    )

    nw.new_node(
        Nodes.MaterialOutput,
        input_kwargs={"Surface": white_noise_texture.outputs["Color"]},
    )


def _replace_shader_with_randcolor(material: bpy.types.Material):
    nt = material.node_tree
    if nt is None:
        return
    logger.debug(f"Replacing shader with randcolor for {material.name}")
    nodes = nt.nodes
    object_info = nodes.new(type="ShaderNodeObjectInfo")
    white_noise_texture = nodes.new(type="ShaderNodeTexWhiteNoise")
    material_output = nodes["Material Output"]
    nt.links.new(object_info.outputs["Random"], white_noise_texture.inputs["Vector"])
    nt.links.new(
        white_noise_texture.outputs["Color"], material_output.inputs["Surface"]
    )


def _remove_volume_shading(material: bpy.types.Material):
    nt = material.node_tree
    if nt is None:
        return
    nw = NodeWrangler(nt)
    for output in nw.find(Nodes.MaterialOutput):
        if "Volume" not in output.inputs:
            continue
        vol_socket = output.inputs["Volume"]
        if len(vol_socket.links) > 0:
            nw.links.remove(vol_socket.links[0])


def _replace_materials_with_flat_shading(obj: bpy.types.Object):
    for i in range(len(obj.material_slots)):
        if obj.material_slots[i] is None or obj.material_slots[i].material is None:
            logger.debug(
                f"Skipping {obj.name} with empty material slot {i}/{len(obj.material_slots)}"
            )
            continue
        try:
            _replace_shader_with_randcolor(obj.material_slots[i].material)
        except Exception as e:
            mat = obj.material_slots[i].material
            raise RuntimeError(
                f"Error in blendergt flat_shading {_replace_shader_with_randcolor.__name__} for "
                f"{obj.name} with material slot {i} {mat.name}: {e}"
            )


def global_flat_shading():
    # Remove all volumes in the scene as they cause noisy depth
    for obj in bpy.context.scene.view_layers["ViewLayer"].objects:
        if "fire_system_type" in obj and obj["fire_system_type"] == "volume":
            continue
        if obj.name.lower() in {"atmosphere", "atmosphere_fine"}:
            bpy.data.objects.remove(obj)
            continue
        if obj.active_material is None:
            continue
        try:
            _remove_volume_shading(obj.active_material)
        except Exception as e:
            mat = obj.active_material
            raise RuntimeError(
                f"Error in blendergt flat_shading {_remove_volume_shading.__name__} for "
                f"{obj.name} with material {mat.name}: {e}"
            )

    bpy.context.view_layer.update()

    # Get rid of all nondiffuse materials. e.g. glass becomes solid, or else we get noisy depth (as of bl3.6 at least)
    for obj in bpy.context.scene.view_layers["ViewLayer"].objects:
        if obj.type != "MESH":
            logger.debug(
                f"{global_flat_shading.__name__} skipping {obj.name} with non-MESH type {obj.type}"
            )
            continue
        obj.hide_viewport = False
        if "fire_system_type" in obj and obj["fire_system_type"] == "gt_mesh":
            obj.hide_viewport = False
            obj.hide_render = False
        if (
            not hasattr(obj, "material_slots")
            or obj.material_slots is None
            or len(obj.material_slots) == 0
        ):
            logger.debug(
                f"{global_flat_shading.__name__} skipping {obj.name} with no material slots"
            )
            continue
        _replace_materials_with_flat_shading(obj)

    nw = NodeWrangler(bpy.data.worlds["World"].node_tree)
    for link in nw.links:
        nw.links.remove(link)


def postprocess_blendergt_outputs(frames_folder, output_stem, camera):
    # Save flow visualization
    flow_dst_path = frames_folder / f"Vector{output_stem}.exr"
    flow_array = load_flow(flow_dst_path)
    np.save(flow_dst_path.with_name(f"Flow{output_stem}.npy"), flow_array)

    flow_color = colorize_flow(flow_array)
    if flow_color is not None:
        imwrite(
            flow_dst_path.with_name(f"Flow{output_stem}.png"),
            flow_color,
        )
        flow_dst_path.unlink()

    # Save surface normal visualization
    normal_dst_path = frames_folder / f"Normal{output_stem}.exr"
    normal_array = load_normals(normal_dst_path, camera)
    np.save(flow_dst_path.with_name(f"SurfaceNormal{output_stem}.npy"), normal_array)
    imwrite(
        flow_dst_path.with_name(f"SurfaceNormal{output_stem}.png"),
        colorize_normals(normal_array),
    )
    normal_dst_path.unlink()

    # Save depth visualization
    depth_dst_path = frames_folder / f"Depth{output_stem}.exr"
    depth_array = load_depth(depth_dst_path)
    np.save(flow_dst_path.with_name(f"Depth{output_stem}.npy"), depth_array)
    imwrite(
        depth_dst_path.with_name(f"Depth{output_stem}.png"), colorize_depth(depth_array)
    )
    depth_dst_path.unlink()

    # Save segmentation visualization
    seg_dst_path = frames_folder / f"IndexOB{output_stem}.exr"
    seg_mask_array = load_seg_mask(seg_dst_path)
    np.save(
        flow_dst_path.with_name(f"ObjectSegmentation{output_stem}.npy"), seg_mask_array
    )
    imwrite(
        seg_dst_path.with_name(f"ObjectSegmentation{output_stem}.png"),
        colorize_int_array(seg_mask_array),
    )
    seg_dst_path.unlink()

    # Save unique instances visualization
    uniq_inst_path = frames_folder / f"UniqueInstances{output_stem}.exr"
    uniq_inst_array = load_uniq_inst(uniq_inst_path)
    np.save(
        flow_dst_path.with_name(f"InstanceSegmentation{output_stem}.npy"),
        uniq_inst_array,
    )
    imwrite(
        uniq_inst_path.with_name(f"InstanceSegmentation{output_stem}.png"),
        colorize_int_array(uniq_inst_array),
    )
    uniq_inst_path.unlink()


def postprocess_materialgt_output(frames_folder, output_stem):
    # Save material segmentation visualization if present
    ma_seg_dst_path = frames_folder / f"IndexMA{output_stem}.exr"
    if ma_seg_dst_path.is_file():
        ma_seg_mask_array = load_seg_mask(ma_seg_dst_path)
        np.save(
            ma_seg_dst_path.with_name(f"MaterialSegmentation{output_stem}.npy"),
            ma_seg_mask_array,
        )
        imwrite(
            ma_seg_dst_path.with_name(f"MaterialSegmentation{output_stem}.png"),
            colorize_int_array(ma_seg_mask_array),
        )
        ma_seg_dst_path.unlink()


def configure_compositor(
    frames_folder: Path,
    passes_to_save: list,
    flat_shading: bool,
    output_prefix="",
    save_alpha=False,
    binary_mask_threshold=0.5,
):
    compositor_node_tree = bpy.context.scene.node_tree
    nw = NodeWrangler(compositor_node_tree)

    render_layers = nw.new_node(Nodes.RenderLayers)
    final_image_denoised = compositor_postprocessing(
        nw, source=render_layers.outputs["Image"]
    )

    final_image_noisy = (
        compositor_postprocessing(
            nw, source=render_layers.outputs["Noisy Image"], show=False
        )
        if bpy.context.scene.cycles.use_denoising
        else None
    )

    return configure_compositor_output(
        nw,
        frames_folder,
        image_denoised=final_image_denoised,
        image_noisy=final_image_noisy,
        passes_to_save=passes_to_save,
        saving_ground_truth=flat_shading,
        output_prefix=output_prefix,
        save_alpha=save_alpha,
        binary_mask_threshold=binary_mask_threshold,
    )


def _find_subject_objects(subject_collection_names, subject_name_patterns):
    subject_names = set()
    for collection_name in subject_collection_names:
        collection = bpy.data.collections.get(collection_name)
        if collection is None:
            logger.warning(f"Could not find subject collection '{collection_name}'")
            continue
        subject_names |= {obj.name for obj in collection.all_objects}

    if subject_name_patterns:
        for obj in bpy.data.objects:
            if any(fnmatch(obj.name, p) for p in subject_name_patterns):
                subject_names.add(obj.name)

    return sorted(subject_names)


def _set_subject_visibility(subject_names, subject_mode):
    prior_visibility = {obj.name: obj.hide_render for obj in bpy.data.objects}
    subject_set = set(subject_names)
    for obj in bpy.data.objects:
        is_subject = obj.name in subject_set
        if subject_mode == "foreground_only":
            obj.hide_render = not is_subject
        elif subject_mode == "background_only":
            obj.hide_render = is_subject
        else:
            obj.hide_render = prior_visibility[obj.name]
    return prior_visibility


def _restore_visibility(prior_visibility):
    for name, hide_render in prior_visibility.items():
        if name in bpy.data.objects:
            bpy.data.objects[name].hide_render = hide_render


def _write_render_manifest(frames_folder, suffix, entries):
    (frames_folder / f"RenderManifest{suffix}.json").write_text(
        json.dumps(entries, indent=2)
    )


def _postprocess_subject_outputs(frames_folder, suffix, trimap_radius):
    alpha_path = frames_folder / f"ForegroundAlphaMatte{suffix}.png"
    binary_path = frames_folder / f"ForegroundBinaryMask{suffix}.png"
    if not alpha_path.exists() or not binary_path.exists():
        return

    alpha = iio.imread(alpha_path)
    if alpha.ndim == 3:
        alpha = alpha[..., 0]
    alpha = alpha.astype(np.uint8)
    imwrite(alpha_path, alpha)

    binary = (alpha > 127).astype(np.uint8) * 255
    imwrite(binary_path, binary)

    if trimap_radius <= 0:
        trimap = np.where(binary > 0, 255, 0).astype(np.uint8)
    else:
        import cv2

        kernel = np.ones((2 * trimap_radius + 1, 2 * trimap_radius + 1), np.uint8)
        fg = cv2.erode(binary, kernel)
        bg = cv2.erode(255 - binary, kernel)
        trimap = np.full(binary.shape, 128, dtype=np.uint8)
        trimap[bg > 0] = 0
        trimap[fg > 0] = 255
    imwrite(frames_folder / f"ForegroundTrimap{suffix}.png", trimap)


def _unlink_material_displacement_output(material: bpy.types.Material):
    if material.node_tree is None:
        return
    nw = NodeWrangler(material.node_tree)
    material_outputs = nw.find(Nodes.MaterialOutput)
    for output_node in material_outputs:
        if "Displacement" not in output_node.inputs:
            continue
        displacement_input = output_node.inputs["Displacement"]
        for link in displacement_input.links:
            logger.debug(
                f"{_unlink_material_displacement_output.__name__} removing {link} to {output_node.name} in {material.name}"
            )
            nw.links.remove(link)


@gin.configurable
def set_displacement_mode(
    displacement_mode: Literal["DISPLACEMENT", "BUMP", "BOTH", "NONE"] = "DISPLACEMENT",
):
    match displacement_mode:
        case "NONE":
            for material in bpy.data.materials:
                _unlink_material_displacement_output(material)
        case "DISPLACEMENT" | "BUMP" | "BOTH":
            for material in bpy.data.materials:
                set_geometry_option(material, displacement_mode)
        case _:
            raise ValueError(f"Invalid displacement mode: {displacement_mode}")


@gin.configurable
def render_image(
    camera: bpy.types.Object,
    frames_folder,
    passes_to_save,
    render_resolution_override=None,
    excludes=[],
    use_dof=False,
    dof_aperture_fstop=2.8,
    flat_shading=False,
    override_num_samples=None,
    render_subject_background_pairs=False,
    subject_collection_names=(),
    subject_name_patterns=("subject*",),
    trimap_radius=4,
    binary_mask_threshold=0.5,
):
    tic = time.time()

    for exclude in excludes:
        bpy.data.objects[exclude].hide_render = True

    init.configure_cycles_devices()
    set_displacement_mode()

    tmp_dir = frames_folder.parent.resolve() / "tmp"
    tmp_dir.mkdir(exist_ok=True)
    bpy.context.scene.render.filepath = f"{tmp_dir}{os.sep}"

    camrig_id, subcam_id = cam_util.get_id(camera)

    if override_num_samples is not None:  # usually used for GT
        bpy.context.scene.cycles.samples = override_num_samples

    if flat_shading:
        with Timer("Set object indices"):
            object_data = set_pass_indices()
            json_object = json.dumps(object_data, indent=4)
            first_frame = bpy.context.scene.frame_start
            suffix = get_suffix(
                dict(
                    cam_rig=camrig_id,
                    resample=0,
                    frame=first_frame,
                    subcam=subcam_id,
                )
            )
            (frames_folder / f"Objects{suffix}.json").write_text(json_object)

        with Timer("Flat Shading"):
            global_flat_shading()
    else:
        segment_materials = "material_index" in (x[0] for x in passes_to_save)
        if segment_materials:
            with Timer("Set material indices"):
                material_data = set_material_pass_indices()
                json_object = json.dumps(material_data, indent=4)
                first_frame = bpy.context.scene.frame_start
                suffix = get_suffix(
                    dict(
                        cam_rig=camrig_id,
                        resample=0,
                        frame=first_frame,
                        subcam=subcam_id,
                    )
                )
                (frames_folder / f"Materials{suffix}.json").write_text(json_object)

    if not bpy.context.scene.use_nodes:
        bpy.context.scene.use_nodes = True

    indices = dict(cam_rig=camrig_id, resample=0, subcam=subcam_id)

    def run_render_pass(output_prefix="", save_alpha=False):
        compositor_node_tree = bpy.context.scene.node_tree
        compositor_node_tree.nodes.clear()
        compositor_node_tree.links.clear()
        file_slot_nodes = configure_compositor(
            frames_folder,
            passes_to_save,
            flat_shading,
            output_prefix=output_prefix,
            save_alpha=save_alpha,
            binary_mask_threshold=binary_mask_threshold,
        )
        fileslot_suffix = get_suffix({"frame": "####", **indices})
        for file_slot in file_slot_nodes:
            file_slot.path = f"{file_slot.path}{fileslot_suffix}"
        with Timer(f"Actual rendering ({output_prefix or 'Main'})"):
            bpy.ops.render.render(animation=True)

    if use_dof == "IF_TARGET_SET":
        use_dof = camera.data.dof.focus_object is not None
    elif use_dof is not None:
        camera.data.dof.use_dof = use_dof
        camera.data.dof.aperture_fstop = dof_aperture_fstop

    if render_resolution_override is not None:
        bpy.context.scene.render.resolution_x = render_resolution_override[0]
        bpy.context.scene.render.resolution_y = render_resolution_override[1]

    bpy.context.scene.camera = camera

    render_manifest = []
    if render_subject_background_pairs and not flat_shading:
        subject_names = _find_subject_objects(
            subject_collection_names=subject_collection_names,
            subject_name_patterns=subject_name_patterns,
        )
        if subject_names:
            vis = _set_subject_visibility(subject_names, "foreground_only")
            run_render_pass(output_prefix="Foreground", save_alpha=True)
            _restore_visibility(vis)

            vis = _set_subject_visibility(subject_names, "background_only")
            run_render_pass(output_prefix="Background", save_alpha=False)
            _restore_visibility(vis)
        else:
            logger.warning(
                "render_subject_background_pairs=True but no subject objects were found; skipping pair renders"
            )

    run_render_pass()

    with Timer("Post Processing"):
        for frame in range(
            bpy.context.scene.frame_start, bpy.context.scene.frame_end + 1
        ):
            if flat_shading:
                bpy.context.scene.frame_set(frame)
                suffix = get_suffix(dict(frame=frame, **indices))
                postprocess_blendergt_outputs(frames_folder, suffix, camera)
            else:
                cam_util.save_camera_parameters(
                    camera,
                    output_folder=frames_folder,
                    frame=frame,
                )
                bpy.context.scene.frame_set(frame)
                suffix = get_suffix(dict(frame=frame, **indices))
                postprocess_materialgt_output(frames_folder, suffix)

                frame_manifest = {
                    "frame_suffix": suffix,
                    "main_rgb": f"Image{suffix}.png",
                }
                if render_subject_background_pairs:
                    _postprocess_subject_outputs(frames_folder, suffix, trimap_radius)
                    fg_image = frames_folder / f"ForegroundImage{suffix}.png"
                    bg_image = frames_folder / f"BackgroundImage{suffix}.png"
                    alpha = frames_folder / f"ForegroundAlphaMatte{suffix}.png"
                    binary = frames_folder / f"ForegroundBinaryMask{suffix}.png"
                    trimap = frames_folder / f"ForegroundTrimap{suffix}.png"
                    if fg_image.exists() and bg_image.exists():
                        frame_manifest.update(
                            {
                                "foreground_rgb": fg_image.name,
                                "background_rgb": bg_image.name,
                                "alpha_matte": alpha.name,
                                "binary_mask": binary.name,
                                "trimap": trimap.name,
                            }
                        )
                render_manifest.append(frame_manifest)

                _write_render_manifest(frames_folder, suffix, frame_manifest)

    if render_manifest and not flat_shading:
        (frames_folder / "RenderManifest.json").write_text(
            json.dumps(render_manifest, indent=2)
        )

    for file in tmp_dir.glob("*.png"):
        file.unlink()

    reorganize_old_framesfolder(frames_folder)

    logger.info(f"rendering time: {time.time() - tic}")
