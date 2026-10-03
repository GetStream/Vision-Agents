"""
Renders one scene script to a PNG with Blender.

It runs wherever `bpy` is installed, which is the Daytona sandbox when the agent uses it:

    python render.py scene.py render.png 960 540 32

The scene script starts from an empty file with a dark world. Whatever it leaves out of
a camera and a light is added here, so a script that only builds objects still renders
something worth looking at.
"""

import math
import sys

import bpy
from mathutils import Vector


def main(scene: str, output: str, width: int, height: int, samples: int) -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    # An empty file has no world, and scripts reach for scene.world as if it had one.
    bpy.context.scene.world = bpy.data.worlds.new("World")
    bpy.context.scene.world.color = (0.05, 0.05, 0.06)
    with open(scene) as source:
        exec(compile(source.read(), scene, "exec"), {"__name__": "__main__"})

    current = bpy.context.scene
    _light(current)
    _frame(current, width / height)

    current.render.engine = "CYCLES"
    current.cycles.device = "CPU"
    current.cycles.samples = samples
    current.cycles.use_denoising = True
    current.render.resolution_x = width
    current.render.resolution_y = height
    current.render.resolution_percentage = 100
    current.render.image_settings.file_format = "PNG"
    current.render.filepath = output
    bpy.ops.render.render(write_still=True)


def _light(scene: bpy.types.Scene) -> None:
    if not any(obj.type == "LIGHT" for obj in scene.objects):
        sun = bpy.data.objects.new("Sun", bpy.data.lights.new("Sun", type="SUN"))
        sun.data.energy = 3.0
        sun.rotation_euler = (math.radians(50), 0.0, math.radians(30))
        scene.collection.objects.link(sun)


def _frame(scene: bpy.types.Scene, aspect: float) -> None:
    """Point a camera at the scene's subject when the script did not place one.

    Flat objects such as a floor or a sea are left out, since framing them would make
    whatever stands on them a speck in the middle.
    """
    if scene.camera is not None:
        return
    boxes = [
        [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
        for obj in scene.objects
        if obj.type in {"MESH", "CURVE", "SURFACE", "META", "FONT"}
    ]
    solid = [
        box for box in boxes if max(c.z for c in box) - min(c.z for c in box) > 1e-3
    ]
    corners = [corner for box in solid or boxes for corner in box]
    if not corners:
        corners = [Vector((0.0, 0.0, 0.0))]
    low = Vector(tuple(min(c[i] for c in corners) for i in range(3)))
    high = Vector(tuple(max(c[i] for c in corners) for i in range(3)))
    centre = (low + high) / 2
    radius = max((high - low).length / 2, 0.5)

    camera = bpy.data.objects.new("Camera", bpy.data.cameras.new("Camera"))
    scene.collection.objects.link(camera)
    fov = camera.data.angle if aspect >= 1 else camera.data.angle * aspect
    direction = Vector((1.0, -1.2, 0.7)).normalized()
    camera.location = centre + direction * (radius / math.sin(fov / 2)) * 1.25
    camera.rotation_euler = (-direction).to_track_quat("-Z", "Y").to_euler()
    camera.data.clip_end = max(100.0, radius * 20)
    scene.camera = camera


if __name__ == "__main__":
    scene_path, output_path, width, height, samples = sys.argv[1:6]
    main(scene_path, output_path, int(width), int(height), int(samples))
