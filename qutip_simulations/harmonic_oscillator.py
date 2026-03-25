"""
harmonic_oscillator.py
----------------------
Driven and damped quantum harmonic oscillator simulation using QuTiP.

Provides :class:`HarmonicOscillatorSimulation` which models a single bosonic
mode with optional coherent driving and photon loss.

Examples
--------
>>> from qutip_simulations.harmonic_oscillator import HarmonicOscillatorSimulation
>>> sim = HarmonicOscillatorSimulation(omega=1.0, n_fock=20, kappa=0.1)
>>> result = sim.run(tmax=20.0, steps=200)
"""

from __future__ import annotations

import numpy as np
import qutip as qt


class HarmonicOscillatorSimulation:
    """Simulate a quantum harmonic oscillator with optional driving and damping.

    The Hamiltonian is

        H = ω a†a + ε(t)(a + a†)

    where *ε(t)* is a coherent drive.  Photon loss is modelled by the
    Lindblad collapse operator ``√κ · a``.

    Parameters
    ----------
    omega : float
        Oscillator frequency.
    n_fock : int
        Fock-space truncation (number of levels).
    drive_amplitude : float, optional
        Amplitude of the coherent driving field.  Default is 0.
    drive_frequency : float, optional
        Frequency of the driving field.  Default is 0.
    kappa : float, optional
        Photon loss rate.  Default is 0.
    initial_state : :class:`qutip.Qobj` or None
        Initial state.  Defaults to the vacuum state ``|0⟩``.
    """

    def __init__(
        self,
        omega: float,
        n_fock: int = 20,
        drive_amplitude: float = 0.0,
        drive_frequency: float = 0.0,
        kappa: float = 0.0,
        initial_state: qt.Qobj | None = None,
    ) -> None:
        self.omega = omega
        self.n_fock = n_fock
        self.drive_amplitude = drive_amplitude
        self.drive_frequency = drive_frequency
        self.kappa = kappa
        self.initial_state = initial_state if initial_state is not None else qt.basis(n_fock, 0)

    def _build_hamiltonian(self) -> list | qt.Qobj:
        a = qt.destroy(self.n_fock)
        H0 = self.omega * a.dag() * a
        if self.drive_amplitude == 0.0:
            return H0
        H1 = self.drive_amplitude * (a + a.dag())
        return [H0, [H1, lambda t, args: np.cos(self.drive_frequency * t)]]

    def _build_collapse_operators(self) -> list:
        if self.kappa > 0.0:
            a = qt.destroy(self.n_fock)
            return [np.sqrt(self.kappa) * a]
        return []

    def run(
        self,
        tmax: float,
        steps: int = 100,
        observables: list[qt.Qobj] | None = None,
    ) -> qt.solver.Result:
        """Evolve the oscillator and return the solver result.

        Parameters
        ----------
        tmax : float
            Total evolution time.
        steps : int
            Number of time steps to record.
        observables : list of :class:`qutip.Qobj`, optional
            Operators whose expectation values are tracked.  Defaults to
            the number operator ``n = a†a``.

        Returns
        -------
        :class:`qutip.solver.Result`
        """
        a = qt.destroy(self.n_fock)
        if observables is None:
            observables = [a.dag() * a]

        tlist = np.linspace(0, tmax, steps)
        H = self._build_hamiltonian()
        c_ops = self._build_collapse_operators()

        if c_ops:
            rho0 = qt.ket2dm(self.initial_state) if self.initial_state.type == "ket" else self.initial_state
            return qt.mesolve(H, rho0, tlist, c_ops, observables)
        return qt.sesolve(H, self.initial_state, tlist, observables)
