"""Pure LEED metric calculations.  Arrays in, numbers out.

Nothing in this package opens a file or runs a subprocess, so every LEED rule
can be tested in milliseconds against invented data.

This package follows one axis convention and one polarity convention,
everywhere:

1. **Axis order.** Every array is point-major. A 2-D array is
   ``(points, times)`` — hours for Option 1, clock times for Option 2. A 3-D
   array is ``(groups, points, times)``. The point axis always precedes the
   time axis; a group axis, when present, comes first. Every reduction over
   time is ``axis=-1``.
2. **Polarity.** A function named ``*_pass`` returns True for a point that
   satisfies the criterion. A function named ``*_exceeds`` returns True for
   a point that is over a limit. No other polarity naming is used.
3. **Units.** Every illuminance argument is **lux**. Layer 4 converts
   Radiance RGB to lux and nothing above layer 4 ever sees a radiance value.
   This is a boundary rule, not a preference.
"""
