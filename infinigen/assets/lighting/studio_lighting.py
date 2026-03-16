# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

# Authors: OpenAI Assistant

import bpy
from numpy.random import uniform as U

from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler


def _set_emission_world(strength=0.15):
    world = bpy.context.scene.world
    world.use_nodes = True
    nw = NodeWrangler(world.node_tree)
    bg = nw.new_node(
        Nodes.Background,
        input_kwargs={"Color": (0.01, 0.01, 0.01, 1.0), "Strength": strength},
    )
    nw.new_node(Nodes.WorldOutput, input_kwargs={"Surface": bg})


def procedural_backdrop_material(variant="green"):
    """Create a smooth cyclorama-style backdrop material for chroma key shoots."""

    palette = {
        "green": ((0.16, 0.72, 0.25, 1.0), (0.11, 0.58, 0.20, 1.0)),
        "blue": ((0.14, 0.31, 0.84, 1.0), (0.10, 0.22, 0.66, 1.0)),
    }
    c1, c2 = palette.get(variant, palette["green"])

    mat = bpy.data.materials.new(name=f"StudioBackdrop_{variant.title()}")
    mat.use_nodes = True
    nw = NodeWrangler(mat.node_tree)

    texcoord = nw.new_node(Nodes.TextureCoord)
    mapping = nw.new_node(
        Nodes.Mapping,
        input_kwargs={"Scale": (1.2, 1.2, 1.2), "Vector": texcoord.outputs["Object"]},
    )
    noise = nw.new_node(
        Nodes.NoiseTexture,
        input_kwargs={"Scale": 2.5, "Detail": 8.0, "Roughness": 0.25, "Vector": mapping},
    )
    ramp = nw.new_node(Nodes.ColorRamp, input_kwargs={"Fac": noise.outputs["Fac"]})
    ramp.color_ramp.elements[0].position = 0.35
    ramp.color_ramp.elements[0].color = c1
    ramp.color_ramp.elements[1].position = 0.85
    ramp.color_ramp.elements[1].color = c2
    shader = nw.new_node(
        Nodes.PrincipledBSDF,
        input_kwargs={
            "Base Color": ramp.outputs["Color"],
            "Roughness": 0.6,
            "Specular IOR Level": 0.05,
        },
    )
    nw.new_node(Nodes.MaterialOutput, input_kwargs={"Surface": shader})

    return mat


def add_controlled_studio_lighting(key_energy=1200, fill_ratio=0.45, rim_ratio=0.65):
    """Create a stable, low-variance key/fill/rim setup suitable for VFX stage renders."""

    _set_emission_world(strength=0.08)

    setup = [
        ("AREA", (3.8, -3.4, 3.8), key_energy),
        ("AREA", (3.1, 3.2, 2.8), key_energy * fill_ratio),
        ("SPOT", (-3.8, 1.0, 3.5), key_energy * rim_ratio),
        ("POINT", (0.0, 0.0, 4.4), key_energy * U(0.10, 0.18)),
    ]

    for light_type, location, energy in setup:
        bpy.ops.object.light_add(type=light_type, location=location)
        light = bpy.context.active_object
        light.data.energy = energy
        if light_type == "AREA":
            light.data.shape = "RECTANGLE"
            light.data.size = 1.2
            light.data.size_y = 0.8
        if light_type == "SPOT":
            light.data.spot_size = 1.0
            light.data.spot_blend = 0.2

    return [obj for obj in bpy.context.scene.objects if obj.type == "LIGHT"]
