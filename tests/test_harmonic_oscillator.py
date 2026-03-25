"""Tests for the HarmonicOscillatorSimulation class."""

import numpy as np
import pytest
import qutip as qt

from qutip_simulations.harmonic_oscillator import HarmonicOscillatorSimulation


class TestHarmonicOscillatorSimulation:
    def test_default_initial_state_is_vacuum(self):
        sim = HarmonicOscillatorSimulation(omega=1.0, n_fock=10)
        assert sim.initial_state == qt.basis(10, 0)

    def test_vacuum_photon_number_is_zero(self):
        sim = HarmonicOscillatorSimulation(omega=1.0, n_fock=10)
        result = sim.run(tmax=5.0, steps=50)
        n_expect = result.expect[0]
        assert np.allclose(n_expect, 0.0, atol=1e-6)

    def test_run_returns_correct_number_of_time_steps(self):
        sim = HarmonicOscillatorSimulation(omega=1.0, n_fock=10)
        result = sim.run(tmax=4.0, steps=40)
        assert len(result.times) == 40

    def test_fock_state_photon_number_constant(self):
        n_photons = 3
        fock3 = qt.basis(10, n_photons)
        sim = HarmonicOscillatorSimulation(omega=1.0, n_fock=10, initial_state=fock3)
        result = sim.run(tmax=5.0, steps=50)
        n_expect = result.expect[0]
        assert np.allclose(n_expect, n_photons, atol=1e-6)

    def test_damped_oscillator_photon_number_decays(self):
        fock3 = qt.basis(10, 3)
        sim = HarmonicOscillatorSimulation(omega=1.0, n_fock=10, kappa=0.5, initial_state=fock3)
        result = sim.run(tmax=10.0, steps=200)
        n_expect = result.expect[0]
        assert n_expect[-1] < n_expect[0]

    def test_custom_fock_dimension(self):
        sim = HarmonicOscillatorSimulation(omega=1.0, n_fock=5)
        assert sim.initial_state.shape == (5, 1)

    def test_custom_observable(self):
        a = qt.destroy(10)
        sim = HarmonicOscillatorSimulation(omega=1.0, n_fock=10)
        result = sim.run(tmax=2.0, steps=20, observables=[a + a.dag()])
        assert len(result.expect) == 1
