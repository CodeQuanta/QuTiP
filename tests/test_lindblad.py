"""Tests for the LindbladSimulation class."""

import numpy as np
import pytest
import qutip as qt

from qutip_simulations.lindblad import LindbladSimulation


class TestLindbladSimulation:
    def _qubit_setup(self, gamma=0.5):
        H = 0.5 * qt.sigmaz()
        c_ops = [np.sqrt(gamma) * qt.sigmam()]
        # basis(2, 0) is the excited state in QuTiP (<σ_z> = +1).
        # σ_- = |1⟩⟨0| decays it to the ground state basis(2, 1).
        rho0 = qt.ket2dm(qt.basis(2, 0))
        return H, c_ops, rho0

    def test_ket_initial_state_converted_to_dm(self):
        H, c_ops, _ = self._qubit_setup()
        psi = qt.basis(2, 0)
        sim = LindbladSimulation(H, c_ops, psi)
        assert sim.initial_state.type == "oper"

    def test_run_returns_result_with_times(self):
        H, c_ops, rho0 = self._qubit_setup()
        sim = LindbladSimulation(H, c_ops, rho0)
        result = sim.run(tmax=5.0, steps=50)
        assert len(result.times) == 50
        assert np.isclose(result.times[-1], 5.0)

    def test_excited_state_decays_to_ground(self):
        H, c_ops, rho0 = self._qubit_setup(gamma=1.0)
        sim = LindbladSimulation(H, c_ops, rho0)
        result = sim.run(tmax=15.0, steps=200, observables=[qt.sigmaz()])
        sz = result.expect[0]
        assert sz[0] > 0.5       # starts in excited state: <σ_z> ~ +1
        assert sz[-1] < -0.5     # ends in ground state:    <σ_z> ~ -1

    def test_run_with_no_observables(self):
        H, c_ops, rho0 = self._qubit_setup()
        sim = LindbladSimulation(H, c_ops, rho0)
        result = sim.run(tmax=2.0, steps=20)
        assert result.expect == []

    def test_steady_state_ground_state(self):
        H = 0.5 * qt.sigmaz()
        c_ops = [qt.sigmam()]  # strong decay
        rho0 = qt.ket2dm(qt.basis(2, 0))  # excited state
        sim = LindbladSimulation(H, c_ops, rho0)
        ss = sim.steady_state()
        # Ground state population ss[1, 1] (basis(2, 1)) should be ~1
        assert abs(ss[1, 1]) > 0.99

    def test_trace_preserved(self):
        H, c_ops, rho0 = self._qubit_setup()
        sim = LindbladSimulation(H, c_ops, rho0)
        result = sim.run(tmax=5.0, steps=50)
        for state in result.states:
            assert np.isclose(state.tr(), 1.0, atol=1e-6)
