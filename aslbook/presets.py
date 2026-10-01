"""Constants of the simulated brain, of arterial blood, and of the book's reference protocol.

The tissue values are those of the ASLDRO ``hrgt_icbm_2009a_nls_3t`` phantom that aslscan
simulates: one constant per tissue class, which is what makes the phantom's answer key exact
and what makes the toy tier a per-class computation. The blood constants are the aslscan
defaults: the T1 of blood, the partition coefficient and the labeling efficiencies are the
values the ASL white paper (Alsop et al. 2015) recommends at 3 T; the T2 of blood (165 ms) is
the simulator's own default, not a white-paper value.
The protocol constants are the book's reference PCASL acquisition, chosen to follow the white
paper's 2D recommendations within what the simulator models. Seconds and mm unless the name
says otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass

#: Gyromagnetic ratio of the proton over 2 pi (MHz/T) and the field strength of the phantom (T).
GAMMA_BAR_MHZ_PER_T = 42.577
B0_T = 3.0


@dataclass(frozen=True)
class Tissue:
    """One tissue class of the phantom: its label index and its constants."""

    name: str
    label: int
    perfusion: float  #: cerebral blood flow, ml/100 g/min
    att: float  #: arterial transit time, s (CSF carries a 1000 s sentinel: never perfused)
    t1: float  #: s
    t2: float  #: s
    t2star: float  #: s
    m0: float  #: equilibrium magnetization, arbitrary units (same units as the M0 scan)

    @property
    def t2prime(self) -> float:
        """T2' from 1/T2* = 1/T2 + 1/T2', in s."""
        return 1.0 / (1.0 / self.t2star - 1.0 / self.t2)


#: The three tissue classes, keyed by the short name used throughout the book.
TISSUES: dict[str, Tissue] = {
    "GM": Tissue("GM", 1, perfusion=60.0, att=0.8, t1=1.33, t2=0.080, t2star=0.066, m0=74.62188),
    "WM": Tissue("WM", 2, perfusion=20.0, att=1.2, t1=0.83, t2=0.110, t2star=0.053, m0=64.72388),
    "CSF": Tissue("CSF", 3, perfusion=0.0, att=1000.0, t1=3.0, t2=0.300, t2star=0.200, m0=68.04559),
}

#: Label index -> tissue, for the phantom's ``dseg`` maps.
BY_LABEL: dict[int, Tissue] = {t.label: t for t in TISSUES.values()}

# Arterial blood at 3 T (aslscan defaults, the white paper's recommendations).
T1_BLOOD = 1.65  #: s
T2_BLOOD = 0.165  #: s
LAMBDA = 0.9  #: blood-brain partition coefficient of water, ml/g
#: Labeling efficiency by labeling type: aslscan's per-type defaults. The book's CASL dataset
#: overrides CASL to :data:`ALPHA_CASL_AMPLITUDE_MODULATED` through the overlay, so pass
#: ``alpha=`` explicitly when evaluating the kinetic model for that dataset.
ALPHA = {"PCASL": 0.85, "CASL": 0.85, "PASL": 0.98}
ALPHA_CASL_AMPLITUDE_MODULATED = 0.68  #: the efficiency of CASL with an amplitude-modulated control


# The reference protocol: 2D PCASL following the white paper within what aslscan simulates.
@dataclass(frozen=True)
class ReferenceProtocol:
    labeling_duration: float = 1.8  #: s
    post_labeling_delay: float = 1.8  #: s
    repetition_time: float = 4.5  #: s, must hold LD + PLD + the whole slice readout
    echo_time: float = 0.012  #: s
    total_readout_time: float = 0.02  #: s, the EPI train's effective duration
    voxel_mm: tuple[float, float, float] = (3.5, 3.5, 5.0)
    matrix: tuple[int, int] = (64, 68)  #: in-plane acquisition matrix
    n_slices: int = 20
    slice_spacing: float = 0.04  #: s between consecutive 2D slices
    n_pairs: int = 30  #: control-label pairs
    m0_repetition_time: float = 8.0  #: s, the separate M0 scan
    signal_scale: float = 100.0  #: aslscan's intensity scale
    noise_variance: float = 1600.0  #: per-component image noise variance at that scale (sigma 40)

    @property
    def slice_timing(self) -> list[float]:
        return [round(i * self.slice_spacing, 6) for i in range(self.n_slices)]

    @property
    def readout_duration(self) -> float:
        """From the first to the last slice's excitation, s."""
        return self.slice_spacing * (self.n_slices - 1)


REFERENCE = ReferenceProtocol()

#: Signal-to-noise ratio of the reference protocol's gray matter control image, for the text.
#: (signal ~ 100 * 74.6 * (1 - exp(-4.5/1.33)) * exp(-0.012/0.08) ~ 6200; sigma 40.)
REFERENCE_GM_SNR = 155.0
