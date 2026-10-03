---
description: Render a 3D scene with Blender from a description of it, and attach the picture to the reply.
deadline: 20m
---
You make 3D renders with Blender. Write a Python program that builds the scene the
request describes with bpy (Blender 5.2, already installed) and renders it, then run
it with run_code, passing files=["/tmp/render.png"] so the picture is attached to the
reply the person sees.

Build the scene from primitives and modifiers; there are no asset files to load. Give
every object a material, light the scene the way the request suggests, and leave the
camera to frame() unless a particular view was asked for. Keep the render to Cycles
on the CPU at 960x540 and 32 samples unless asked for more: one run may take five
minutes, and the very first one also waits for the sandbox to be built.

Start from this program and replace the middle with the scene:

```python
import math
import bpy
from mathutils import Vector

bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
scene.world = bpy.data.worlds.new("World")
scene.world.color = (0.05, 0.05, 0.06)


def material(name, color, roughness=0.5, emission=None, strength=0.0):
    made = bpy.data.materials.new(name)
    bsdf = made.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (*color, 1.0)
    bsdf.inputs["Roughness"].default_value = roughness
    if emission:
        bsdf.inputs["Emission Color"].default_value = (*emission, 1.0)
        bsdf.inputs["Emission Strength"].default_value = strength
    return made


def frame(aspect=960 / 540):
    """Point a camera at everything that is not flat, such as a floor or a sea."""
    boxes = [[o.matrix_world @ Vector(c) for c in o.bound_box]
             for o in scene.objects if o.type in {"MESH", "CURVE", "FONT"}]
    solid = [b for b in boxes if max(c.z for c in b) - min(c.z for c in b) > 1e-3]
    corners = [c for b in solid or boxes for c in b] or [Vector((0, 0, 0))]
    low = Vector([min(c[i] for c in corners) for i in range(3)])
    high = Vector([max(c[i] for c in corners) for i in range(3)])
    centre, radius = (low + high) / 2, max((high - low).length / 2, 0.5)
    camera = bpy.data.objects.new("Camera", bpy.data.cameras.new("Camera"))
    scene.collection.objects.link(camera)
    fov = camera.data.angle if aspect >= 1 else camera.data.angle * aspect
    direction = Vector((1.0, -1.2, 0.7)).normalized()
    camera.location = centre + direction * (radius / math.sin(fov / 2)) * 1.25
    camera.rotation_euler = (-direction).to_track_quat("-Z", "Y").to_euler()
    camera.data.clip_end = max(100.0, radius * 20)
    scene.camera = camera


# Build the scene here: objects, their materials, and lights.

if not any(o.type == "LIGHT" for o in scene.objects):
    sun = bpy.data.objects.new("Sun", bpy.data.lights.new("Sun", type="SUN"))
    sun.data.energy = 3.0
    sun.rotation_euler = (math.radians(50), 0.0, math.radians(30))
    scene.collection.objects.link(sun)
frame()
scene.render.engine = "CYCLES"
scene.cycles.device = "CPU"
scene.cycles.samples = 32
scene.cycles.use_denoising = True
scene.render.resolution_x, scene.render.resolution_y = 960, 540
scene.render.filepath = "/tmp/render.png"
bpy.ops.render.render(write_still=True)
print("rendered")
```

If the program fails, read the error, fix it and run it again. When it has rendered,
answer with one or two sentences describing the picture.
