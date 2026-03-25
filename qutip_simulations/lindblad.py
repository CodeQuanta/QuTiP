"""
lindblad.py
-----------
General-purpose Lindblad master-equation simulation wrapper.

Provides :class:`LindbladSimulation` which accepts an arbitrary Hamiltonian
and a list of collapse operators, making it straightforward to model any
Markovian open quantum system.

Examples
--------
>>> import qutip as qt
>>> from qutip_simulations.lindblad import LindbladSimulation
>>> H = 0.5 * qt.sigmaz()
>>> c_ops = [0.1 * qt.sigmam()]
>>> rho0 = qt.ket2dm(qt.basis(2, 1))
>>> sim = LindbladSimulation(H, c_ops, rho0)
>>> result = sim.run(tmax=10.0, steps=100)
"""

from __future__ import annotations

import numpy as np
import qutip as qt


class LindbladSimulation:
    """Solve the Lindblad master equation for an arbitrary open quantum system.

    Parameters
    ----------
    hamiltonian : :class:`qutip.Qobj` or list
        System Hamiltonian.  Can be a static ``Qobj`` or a time-dependent
        list in QuTiP's ``[H0, [H1, coeff], ...]`` format.
    collapse_operators : list of :class:`qutip.Qobj`
        Lindblad collapse operators describing the system-bath coupling.
    initial_state : :class:`qutip.Qobj`
        Initial density matrix (or ket, which is converted automatically).
    """

    def __init__(
        self,
        hamiltonian: qt.Qobj | list,
        collapse_operators: list[qt.Qobj],
        initial_state: qt.Qobj,
    ) -> None:
        self.hamiltonian = hamiltonian
        self.collapse_operators = collapse_operators
        if initial_state.type == "ket":
            self.initial_state = qt.ket2dm(initial_state)
        else:
            self.initial_state = initial_state

    def run(
        self,
        tmax: float,
        steps: int = 100,
        observables: list[qt.Qobj] | None = None,
    ) -> qt.solver.Result:
        """Integrate the Lindblad master equation.

        Parameters
        ----------
        tmax : float
            Total evolution time.
        steps : int
            Number of time steps to record.
        observables : list of :class:`qutip.Qobj`, optional
            Operators whose expectation values are tracked.  Defaults to an
            empty list (only the state trajectory is stored).

        Returns
        -------
        :class:`qutip.solver.Result`
            Result object with ``.times``, ``.states``, and ``.expect``.
        """
        if observables is None:
            observables = []
        tlist = np.linspace(0, tmax, steps)
        return qt.mesolve(
            self.hamiltonian,
            self.initial_state,
            tlist,
            self.collapse_operators,
            observables,
        )

    def steady_state(self) -> qt.Qobj:
        """Compute the steady-state density matrix.

        Returns
        -------
        :class:`qutip.Qobj`
            Steady-state density matrix.
        """
        return qt.steadystate(self.hamiltonian, self.collapse_operators)
