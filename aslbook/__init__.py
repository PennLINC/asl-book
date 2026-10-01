"""Helper package for the arterial spin labeling executable book.

Modules
-------
presets   : tissue, blood, and protocol constants of the simulated brain and the reference protocol
protocols : BIDS ASL sidecars and aslcontext tables for the book's protocols
kinetic   : the Buxton general kinetic model and the longitudinal signal equations, as aslscan evaluates them
grid      : the box resampler that takes the 1 mm phantom to an acquisition grid, as aslscan does
phantom   : the packaged phantom slab (tissue fractions and per-tissue constants) for the toy tier
synth     : toy ASL series built in the page from the packaged phantom (no k-space)
quant     : subtraction, CBF quantification, multi-delay fitting, partial-volume correction, scoring
data      : fetch and load the pre-simulated aslscan datasets (pooch registry, local override)
truth     : the ground-truth maps of a simulated run and the comparison metrics
cookbook  : render the aslscan inputs and commands behind every dataset (Appendix A)
plotting  : house style for slice mosaics, error maps, fit-vs-truth panels
"""

__version__ = "0.1.0.dev0"
