"""
cavity_qed.py
-------------
Demonstrate a damped harmonic oscillator (cavity) using
:class:`~qutip_simulations.harmonic_oscillator.HarmonicOscillatorSimulation`.

Run with::

    python examples/cavity_qed.py
"""

import matplotlib.pyplot as plt
import qutip as qt

from qutip_simulations.harmonic_oscillator import HarmonicOscillatorSimulation


def plot_cavity_decay():
    fock3 = qt.basis(20, 3)
    sim = HarmonicOscillatorSimulation(omega=1.0, n_fock=20, kappa=0.2, initial_state=fock3)
    result = sim.run(tmax=15.0, steps=300)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(result.times, result.expect[0], label=r"$\langle n \rangle$")
    ax.set_xlabel("Time")
    ax.set_ylabel("Mean photon number")
    ax.set_title("Cavity Decay (κ = 0.2, initial Fock state |3⟩)")
    ax.legend()
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    fig = plot_cavity_decay()
    plt.show()
