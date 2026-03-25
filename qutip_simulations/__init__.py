"""
qutip_simulations
=================
QuTiP-based physics simulations for open and closed quantum systems.

Modules
-------
qubit
    Single and multi-qubit dynamics under unitary and dissipative evolution.
harmonic_oscillator
    Driven and damped harmonic oscillator simulations.
lindblad
    Lindblad master-equation solver wrappers for open quantum systems.
"""

from .qubit import QubitSimulation
from .harmonic_oscillator import HarmonicOscillatorSimulation
from .lindblad import LindbladSimulation

__all__ = [
    "QubitSimulation",
    "HarmonicOscillatorSimulation",
    "LindbladSimulation",
]
__version__ = "0.1.0"
