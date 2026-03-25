"""
qubit_dynamics.py
-----------------
Demonstrate single-qubit Rabi oscillations and spontaneous emission using
:class:`~qutip_simulations.qubit.QubitSimulation`.

Run with::

    python examples/qubit_dynamics.py
"""

import matplotlib.pyplot as plt
import numpy as np

from qutip_simulations.qubit import QubitSimulation


def plot_rabi_oscillations():
    omega = 1.0
    drive = 0.5
    sim = QubitSimulation(omega=omega, drive_amplitude=drive, drive_frequency=omega)
    result = sim.run(tmax=20.0, steps=500)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(result.times, result.expect[2], label=r"$\langle\sigma_z\rangle$")
    ax.set_xlabel("Time")
    ax.set_ylabel("Expectation value")
    ax.set_title("Qubit Rabi Oscillations")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_spontaneous_emission():
    import qutip as qt

    psi_excited = qt.basis(2, 0)
    sim = QubitSimulation(omega=1.0, gamma=0.3, initial_state=psi_excited)
    result = sim.run(tmax=15.0, steps=300)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(result.times, result.expect[2], label=r"$\langle\sigma_z\rangle$")
    ax.set_xlabel("Time")
    ax.set_ylabel("Expectation value")
    ax.set_title("Spontaneous Emission (γ = 0.3)")
    ax.legend()
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    fig1 = plot_rabi_oscillations()
    fig2 = plot_spontaneous_emission()
    plt.show()
