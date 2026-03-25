"""
qubit.py
--------
Single-qubit and two-qubit simulations using QuTiP.

Provides :class:`QubitSimulation` which wraps QuTiP's ``sesolve``
(Schrödinger equation solver) and ``mesolve`` (Lindblad master-equation
solver) to model qubit dynamics driven by an arbitrary Hamiltonian.

Examples
--------
>>> from qutip_simulations.qubit import QubitSimulation
>>> sim = QubitSimulation(omega=1.0)
>>> result = sim.run(tmax=10.0, steps=200)
>>> result.expect  # list of expectation-value arrays
"""

from __future__ import annotations

import numpy as np
import qutip as qt


class QubitSimulation:
    """Simulate a two-level system (qubit) driven by a static or time-dependent Hamiltonian.

    Parameters
    ----------
    omega : float
        Qubit transition frequency (in rad/s or natural units).
    drive_amplitude : float, optional
        Amplitude of a transverse (σ_x) driving field.  Default is 0.
    drive_frequency : float, optional
        Frequency of the transverse driving field.  Default is 0.
    gamma : float, optional
        Spontaneous emission rate (relaxation).  When non-zero a Lindblad
        collapse operator ``√γ · σ_-`` is added.  Default is 0.
    initial_state : :class:`qutip.Qobj` or None
        Initial qubit state.  Defaults to the ground state ``basis(2, 1)``
        (the −1 eigenstate of σ_z for H = ω/2 σ_z).
    """

    def __init__(
        self,
        omega: float,
        drive_amplitude: float = 0.0,
        drive_frequency: float = 0.0,
        gamma: float = 0.0,
        initial_state: qt.Qobj | None = None,
    ) -> None:
        self.omega = omega
        self.drive_amplitude = drive_amplitude
        self.drive_frequency = drive_frequency
        self.gamma = gamma
        # In QuTiP's convention, basis(2, 1) is the lower-energy eigenstate
        # of σ_z (eigenvalue -1) — the physical ground state for H = ω/2 σ_z.
        self.initial_state = initial_state if initial_state is not None else qt.basis(2, 1)

    def _build_hamiltonian(self) -> list | qt.Qobj:
        H0 = 0.5 * self.omega * qt.sigmaz()
        if self.drive_amplitude == 0.0:
            return H0
        H1 = 0.5 * self.drive_amplitude * qt.sigmax()
        return [H0, [H1, lambda t, args: np.cos(self.drive_frequency * t)]]

    def _build_collapse_operators(self) -> list:
        if self.gamma > 0.0:
            return [np.sqrt(self.gamma) * qt.sigmam()]
        return []

    def run(
        self,
        tmax: float,
        steps: int = 100,
        observables: list[qt.Qobj] | None = None,
    ) -> qt.solver.Result:
        """Evolve the qubit and return the solver result.

        Parameters
        ----------
        tmax : float
            Total evolution time.
        steps : int
            Number of time steps to record.
        observables : list of :class:`qutip.Qobj`, optional
            Operators whose expectation values are tracked.  Defaults to
            ``[σ_x, σ_y, σ_z]``.

        Returns
        -------
        :class:`qutip.solver.Result`
            QuTiP solver result object containing ``.times`` and ``.expect``.
        """
        if observables is None:
            observables = [qt.sigmax(), qt.sigmay(), qt.sigmaz()]

        tlist = np.linspace(0, tmax, steps)
        H = self._build_hamiltonian()
        c_ops = self._build_collapse_operators()

        if c_ops:
            rho0 = qt.ket2dm(self.initial_state) if self.initial_state.type == "ket" else self.initial_state
            return qt.mesolve(H, rho0, tlist, c_ops, observables)
        return qt.sesolve(H, self.initial_state, tlist, observables)
