# leed_daylight/deliver/atlas/blender_build.py
"""Build the atlas's ``model.blend`` from ``bundle.npz`` (atlas spec 12).

Runs only inside Blender 4.2 or newer, started by ``engine/blender.py``:

    blender -b --factory-startup --python-exit-code 1 --python blender_build.py -- <bundle.npz> <model.blend.part>

The package never imports this file to run it; it is handed to Blender as a
script. So it imports nothing from the package: the storey codes of the
bundle (``SITE``, ``BELOW``) and the kinds of geometry are repeated here,
and a test checks that they agree with ``deliver/atlas``. ``bpy`` is
imported inside :func:`main`, so importing this module outside Blender
works (the tests do).

What it builds, from the bundle alone:

* top collections ``MODEL`` (walls, floors, ceilings, fixed shading,
  furniture, glass by kind), ``ADDED`` (frames, reveals, skylight wells,
  blinds), ``CONTEXT`` (context and ground; hidden), ``ROOMS`` (each room's
  name as 3D text, the grid points as small tiles) and ``RESULTS`` (one
  object per result layer and storey: a tile per point, coloured through
  a per-corner ``FLOAT_COLOR`` attribute);
* under each top collection, one sub-collection per storey that has
  something in it (``site`` and ``below`` as the bundle names them);
* one view layer for the whole building and one per storey, which excludes
  every other storey's sub-collections (storeys on one level share);
* one material per kind with the kind's colour and opacity; a material that
  is not opaque (the glass) uses ``surface_render_method = "BLENDED"``;
* every 3D view in Material Preview, so the kinds show their colours and
  the result tiles their attribute colours. Only the first result layer is
  shown; the others are hidden in every view layer, one click away.

Colours in the bundle are sRGB bytes; Blender's colours are linear, so each
is converted. The file is saved under the name it is given (``.part``);
the pipeline renames it once Blender has exited 0.
"""
from __future__ import annotations

import sys

#: The oldest Blender this script supports (``surface_render_method``).
MIN_VERSION = (4, 2)
#: The bundle's storey index of context and ground, and of a surface below
#: the lowest level (``deliver/atlas/bundle.SITE``, ``BELOW``).
SITE = -1
BELOW = -2
#: The top collections, in outliner order.
TOPS = ("MODEL", "ADDED", "CONTEXT", "ROOMS", "RESULTS")
#: The top collection of every kind of geometry (``deliver/atlas/colours.KINDS``).
TOP_OF_KIND = {
    "wall": "MODEL", "floor": "MODEL", "ceiling": "MODEL", "fixed_shading": "MODEL", "furniture": "MODEL",
    "glass_window": "MODEL", "glass_curtain_wall": "MODEL", "glass_interior": "MODEL",
    "glass_diffusing": "MODEL", "glass_roof": "MODEL",
    "frame": "ADDED", "reveal": "ADDED", "skylight_well": "ADDED", "blind": "ADDED",
    "context": "CONTEXT", "ground": "CONTEXT",
}
#: The view layer that shows every storey.
WHOLE_BUILDING = "whole building"
#: The per-corner colour attribute of a result object.
COLOUR_ATTRIBUTE = "atlas_colour"
#: The side of a grid tile, as a share of the grid spacing (the plans' dots).
GRID_TILE_SHARE = 0.4
#: How far result tiles and room names sit above their points, in metres.
LIFT_M = 0.005
#: Colours of the grid tiles and the room names (sRGB bytes; invented).
GRID_RGB = (120, 120, 120)
ROOM_RGB = (40, 40, 40)
EXCLUDED_ROOM_RGB = (150, 150, 150)

USAGE = ("usage: blender -b --factory-startup --python-exit-code 1 --python blender_build.py "
         "-- <bundle.npz> <out.blend>")


def script_args(argv: list[str]) -> list[str]:
    """The arguments after ``--`` on Blender's command line."""
    return list(argv[argv.index("--") + 1:]) if "--" in argv else []


def _array(npz, key: str):
    """One array of the bundle, refused by name when it is missing."""
    if key not in npz.files:
        raise KeyError(f"bundle.npz has no {key!r} array; build the atlas again")
    return npz[key]


def _linear(rgb):
    """sRGB bytes (..., 3) to linear floats 0 to 1."""
    import numpy as np

    c = np.asarray(rgb, dtype=np.float64) / 255.0
    # source: IEC 61966-2-1 (sRGB), the transfer function from an sRGB value to linear light
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _storey_name(index: int, storeys: list[str]) -> str:
    if index == SITE:
        return "site"
    if index == BELOW:
        return "below"
    return storeys[index]


def _mesh(bpy, name: str, verts, counts):
    """A mesh of polygons: ``verts`` (M, 3) in drawing order, ``counts``
    (P,) vertices per polygon. Each polygon keeps its own vertices, so
    corner ``i`` is vertex ``i``."""
    import numpy as np

    verts = np.asarray(verts, dtype=np.float32).reshape(-1, 3)
    counts = np.asarray(counts, dtype=np.int32).reshape(-1)
    mesh = bpy.data.meshes.new(name)
    mesh.vertices.add(len(verts))
    mesh.vertices.foreach_set("co", verts.ravel())
    mesh.loops.add(len(verts))
    mesh.loops.foreach_set("vertex_index", np.arange(len(verts), dtype=np.int32))
    mesh.polygons.add(len(counts))
    starts = np.zeros(len(counts), dtype=np.int32)
    if len(counts):
        starts[1:] = np.cumsum(counts)[:-1]
    mesh.polygons.foreach_set("loop_start", starts)
    mesh.update(calc_edges=True)
    mesh.validate()
    return mesh


def _tiles(xyz, side: float, lift: float):
    """Square tiles of ``side`` centred on each point, ``lift`` above it:
    (4N, 3) vertices and (N,) counts."""
    import numpy as np

    xyz = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    h = side / 2.0
    corners = np.array([(-h, -h, lift), (h, -h, lift), (h, h, lift), (-h, h, lift)])
    verts = (xyz[:, None, :] + corners[None, :, :]).reshape(-1, 3)
    return verts, np.full(len(xyz), 4, dtype=np.int32)


def _material(bpy, name: str, rgba):
    """A material of one colour: ``rgba`` sRGB bytes, the alpha its opacity.
    A material that is not opaque is drawn blended."""
    colour = [float(c) for c in _linear(rgba[:3])]
    alpha = float(rgba[3]) / 255.0
    mat = bpy.data.materials.new(name)
    if mat.node_tree is None:
        mat.use_nodes = True
    mat.diffuse_color = (*colour, alpha)
    bsdf = next(n for n in mat.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
    bsdf.inputs["Base Color"].default_value = (*colour, 1.0)
    bsdf.inputs["Alpha"].default_value = alpha
    if alpha < 1.0:
        if hasattr(mat, "surface_render_method"):
            mat.surface_render_method = "BLENDED"
        else:
            mat.blend_method = "BLEND"
    mat.use_backface_culling = False
    return mat


def _attribute_material(bpy, name: str):
    """A material that shows the ``COLOUR_ATTRIBUTE`` of its mesh as it is
    (emission, so the colours match the legend)."""
    mat = bpy.data.materials.new(name)
    if mat.node_tree is None:
        mat.use_nodes = True
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    for node in list(nodes):
        nodes.remove(node)
    output = nodes.new("ShaderNodeOutputMaterial")
    attribute = nodes.new("ShaderNodeVertexColor")
    attribute.layer_name = COLOUR_ATTRIBUTE
    emission = nodes.new("ShaderNodeEmission")
    links.new(attribute.outputs[0], emission.inputs[0])
    links.new(emission.outputs[0], output.inputs[0])
    mat.diffuse_color = (0.8, 0.8, 0.8, 1.0)
    return mat


def _colour_corners(mesh, rgb_per_point) -> None:
    """Every corner of point ``i``'s tile gets point ``i``'s colour."""
    import numpy as np

    linear = _linear(np.asarray(rgb_per_point).reshape(-1, 3))
    rgba = np.concatenate([linear, np.ones((len(linear), 1))], axis=1)
    corners = np.repeat(rgba, 4, axis=0).astype(np.float32)
    attribute = mesh.color_attributes.new(COLOUR_ATTRIBUTE, "FLOAT_COLOR", "CORNER")
    attribute.data.foreach_set("color", corners.ravel())
    mesh.color_attributes.active_color = attribute


def _layer_collection(view_layer, top: str, name: str):
    return view_layer.layer_collection.children[top].children[name]


def main(argv: list[str]) -> None:
    """Build and save the ``.blend``: ``argv`` is ``[bundle.npz, out]``."""
    try:
        import bpy
    except ImportError:
        raise RuntimeError("blender_build.py runs only inside Blender "
                           f"{MIN_VERSION[0]}.{MIN_VERSION[1]} or newer") from None
    import numpy as np

    if len(argv) != 2:
        raise ValueError(USAGE)
    if tuple(bpy.app.version[:2]) < MIN_VERSION:
        raise RuntimeError(f"Blender {bpy.app.version_string} is older than "
                           f"{MIN_VERSION[0]}.{MIN_VERSION[1]}")
    bundle_path, out_path = argv

    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    npz = np.load(bundle_path, allow_pickle=False)
    storeys = [str(s) for s in _array(npz, "storeys")]
    levels = [float(z) for z in _array(npz, "levels")]
    cell = float(_array(npz, "cell_m"))
    work = float(_array(npz, "work_plane_m"))

    tops = {}
    for name in TOPS:
        tops[name] = bpy.data.collections.new(name)
        scene.collection.children.link(tops[name])
    subs = {}

    def sub(top: str, index: int):
        if (top, index) not in subs:
            collection = bpy.data.collections.new(f"{top} {_storey_name(index, storeys)}")
            tops[top].children.link(collection)
            subs[(top, index)] = collection
        return subs[(top, index)]

    def place(obj, top: str, index: int) -> None:
        sub(top, index).objects.link(obj)

    everything = []
    kind_rgba = _array(npz, "kind_rgba")
    for i, kind in enumerate(str(k) for k in _array(npz, "kinds")):
        mat = _material(bpy, f"atlas {kind}", kind_rgba[i])
        verts = _array(npz, f"k_{kind}_v")
        counts = _array(npz, f"k_{kind}_n")
        storey = _array(npz, f"k_{kind}_s")
        everything.append(verts)
        polygon_of_vertex = np.repeat(np.arange(len(counts)), counts)
        for index in sorted({int(s) for s in storey}):
            pick = storey == index
            mesh = _mesh(bpy, f"{kind} {_storey_name(index, storeys)}", verts[pick[polygon_of_vertex]],
                         counts[pick])
            mesh.materials.append(mat)
            place(bpy.data.objects.new(mesh.name, mesh), TOP_OF_KIND[kind], index)

    grid_mat = _material(bpy, "atlas grid", (*GRID_RGB, 255))
    grid_v, grid_s = _array(npz, "grid_v"), _array(npz, "grid_s")
    for index in sorted({int(s) for s in grid_s}):
        verts, counts = _tiles(grid_v[grid_s == index], GRID_TILE_SHARE * cell, 0.0)
        mesh = _mesh(bpy, f"grid {_storey_name(index, storeys)}", verts, counts)
        mesh.materials.append(grid_mat)
        place(bpy.data.objects.new(mesh.name, mesh), "ROOMS", index)

    room_mats = {True: _material(bpy, "atlas room", (*ROOM_RGB, 255)),
                 False: _material(bpy, "atlas room excluded", (*EXCLUDED_ROOM_RGB, 255))}
    room_xyz, room_s = _array(npz, "room_xyz"), _array(npz, "room_s")
    room_included = _array(npz, "room_included")
    for i, name in enumerate(str(r) for r in _array(npz, "room_names")):
        text = bpy.data.curves.new(f"room {name}", type="FONT")
        text.body = name
        text.size = cell
        text.align_x = "CENTER"
        text.align_y = "CENTER"
        text.materials.append(room_mats[bool(room_included[i])])
        obj = bpy.data.objects.new(text.name, text)
        x, y, z = (float(c) for c in room_xyz[i])
        obj.location = (x, y, z + work + LIFT_M)
        place(obj, "ROOMS", int(room_s[i]))

    result_mat = _attribute_material(bpy, "atlas result")
    hidden = []
    titles = [str(t) for t in _array(npz, "layer_titles")]
    for i, key in enumerate(str(k) for k in _array(npz, "layer_keys")):
        xyz, storey = _array(npz, f"l{i}_v"), _array(npz, f"l{i}_s")
        values, rgb = _array(npz, f"l{i}_value"), _array(npz, f"l{i}_rgb")
        for index in sorted({int(s) for s in storey}):
            pick = storey == index
            verts, counts = _tiles(xyz[pick], cell, LIFT_M)
            mesh = _mesh(bpy, f"{key} {_storey_name(index, storeys)}", verts, counts)
            _colour_corners(mesh, rgb[pick])
            value = mesh.attributes.new("value", "FLOAT", "FACE")
            value.data.foreach_set("value", values[pick].astype(np.float32))
            mesh.materials.append(result_mat)
            obj = bpy.data.objects.new(mesh.name, mesh)
            obj["layer"] = key
            obj["title"] = titles[i]
            place(obj, "RESULTS", index)
            if i > 0:
                hidden.append(obj)

    scene.view_layers[0].name = WHOLE_BUILDING
    for si, label in enumerate(storeys):
        view_layer = scene.view_layers.new(f"storey {label}")
        for (top, index), collection in subs.items():
            if index < 0 or levels[index] != levels[si]:
                _layer_collection(view_layer, top, collection.name).exclude = True
    for view_layer in scene.view_layers:
        view_layer.layer_collection.children["CONTEXT"].hide_viewport = True
        for obj in hidden:
            if obj.name in view_layer.objects:
                obj.hide_set(True, view_layer=view_layer)

    corners = np.concatenate([v.reshape(-1, 3) for v in everything]) if everything else grid_v.reshape(-1, 3)
    centre = (corners.min(axis=0) + corners.max(axis=0)) / 2.0 if len(corners) else np.zeros(3)
    size = float(np.linalg.norm(corners.max(axis=0) - corners.min(axis=0))) if len(corners) else 10.0
    scene.display.shading.color_type = "MATERIAL"
    for screen in bpy.data.screens:
        for area in screen.areas:
            for space in area.spaces:
                if space.type != "VIEW_3D":
                    continue
                space.shading.type = "MATERIAL"
                space.shading.color_type = "MATERIAL"
                space.clip_end = max(space.clip_end, 4.0 * size)
                if space.region_3d is not None:
                    space.region_3d.view_location = [float(c) for c in centre]
                    space.region_3d.view_distance = max(1.5 * size, 1.0)

    npz.close()
    bpy.ops.wm.save_as_mainfile(filepath=out_path, check_existing=False)


if __name__ == "__main__":
    main(script_args(sys.argv))
