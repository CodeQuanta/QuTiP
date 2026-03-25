"""Tests for the QubitSimulation class."""

import numpy as np
import pytest
import qutip as qt

from qutip_simulations.qubit import QubitSimulation


class TestQubitSimulation:
    def test_default_initial_state_is_ground(self):
        # basis(2, 1) is the physical ground state in QuTiP (lower-energy eigenstate of σ_z)
        sim = QubitSimulation(omega=1.0)
        assert sim.initial_state == qt.basis(2, 1)

    def test_custom_initial_state(self):
        psi = qt.basis(2, 1)
        sim = QubitSimulation(omega=1.0, initial_state=psi)
        assert sim.initial_state == psi

    def test_run_returns_result_with_times(self):
        sim = QubitSimulation(omega=1.0)
        result = sim.run(tmax=5.0, steps=50)
        assert len(result.times) == 50
        assert np.isclose(result.times[0], 0.0)
        assert np.isclose(result.times[-1], 5.0)

    def test_run_returns_three_expectation_values_by_default(self):
        sim = QubitSimulation(omega=1.0)
        result = sim.run(tmax=5.0, steps=50)
        assert len(result.expect) == 3  # <σ_x>, <σ_y>, <σ_z>

    def test_ground_state_sigmaz_expectation_is_minus_one(self):
        # In QuTiP, basis(2, 1) is the -1 eigenstate of σ_z (physical ground state).
        sim = QubitSimulation(omega=1.0)
        result = sim.run(tmax=2.0, steps=20)
        sz = result.expect[2]  # <σ_z>
        assert np.allclose(sz, -1.0, atol=1e-6)

    def test_dissipative_run_uses_mesolve(self):
        sim = QubitSimulation(omega=1.0, gamma=0.5)
        result = sim.run(tmax=5.0, steps=50)
        assert len(result.times) == 50
        sz = result.expect[2]
        # Excited population should decay toward ground state; <σ_z> -> -1
        assert sz[-1] < sz[0] or np.isclose(sz[-1], -1.0, atol=0.1)

    def test_dissipative_excited_state_decays(self):
        # basis(2, 0) is the excited state in QuTiP (<σ_z> = +1).
        # Collapse operator σ_- = |1⟩⟨0| decays it to the ground state |1⟩.
        psi_excited = qt.basis(2, 0)
        sim = QubitSimulation(omega=1.0, gamma=1.0, initial_state=psi_excited)
        result = sim.run(tmax=10.0, steps=200)
        sz = result.expect[2]
        # Start near +1 (excited), end near -1 (ground)
        assert sz[0] > 0.5
        assert sz[-1] < -0.5

    def test_driven_qubit_rabi_oscillations(self):
        # On resonance (drive_frequency == omega) Rabi oscillations occur
        omega = 1.0
        drive = 0.5
        sim = QubitSimulation(omega=omega, drive_amplitude=drive, drive_frequency=omega)
        result = sim.run(tmax=20.0, steps=500)
        sz = result.expect[2]
        # Expect oscillations: not all values should be -1
        assert not np.allclose(sz, sz[0], atol=0.1)

    def test_custom_observable(self):
        sim = QubitSimulation(omega=1.0)
        result = sim.run(tmax=2.0, steps=20, observables=[qt.num(2)])
        assert len(result.expect) == 1
