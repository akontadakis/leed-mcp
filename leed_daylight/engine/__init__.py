"""Running Radiance and building its inputs (layer 4).

``radiance.py`` and ``blender.py`` are the only modules in this package --
and the only modules in the whole tool -- that ever shell out to an
external process: Radiance for the simulations, Blender for the atlas's
``model.blend``. Layers above 4 work in lux and never see a subprocess or
a Radiance RGB triplet.
"""
