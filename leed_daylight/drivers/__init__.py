# leed_daylight/drivers/__init__.py
"""Complete LEED-credit runs (layer 7/the top).

A driver is the `run` verb's business logic for one credit option, with no
command-line front end of its own -- a CLI module calls it. Every driver
calls the engine only through an injectable seam (its own Protocol,
matching one engine class's public surface), so a driver's own tests never
build the real engine and never execute a Radiance binary.
"""
