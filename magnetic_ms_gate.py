#!/usr/bin/env python3
"""
magnetic_ms_gate.py
-------------------

QuTiP simulation of the **magnetic Mølmer-Sørensen (MS) gate** for
trapped-ion quantum computing.

The gate is driven by an oscillating magnetic-field gradient produced by a
current-carrying wire *meander*.  The gradient couples the ion spin states to
a shared motional mode, creating an entangling XX interaction.

This module provides:

* ``WireMeanderGeometry``   – geometric / electrical parameters of the meander
* ``IonParameters``         – trapped-ion physical parameters
* ``GateParameters``        – gate drive and solver parameters
* ``run_simulation()``      – full QuTiP integration returning populations +
                              gate fidelity
* ``sweep_parameter()``     – scan one numeric parameter vs gate fidelity
* ``MagneticMSGateGUI``     – interactive matplotlib-widgets GUI
* ``main()``                – command-line entry-point (``--gui`` flag)

Physics background
------------------
The effective spin-motion coupling Hamiltonian in the doubly-rotating frame is

    H(t) = ħ Ω_eff (σ_x⁽¹⁾ + σ_x⁽²⁾)(a + a†) cos(δ t)

where

    Ω_eff = η Ω_R            effective coupling rate
    η     = (g_s μ_B / ħ) × (∂B/∂z) × x_zpf / ω_m   (dimensionless)
    δ                        drive detuning from the motional sideband

After one gate time  T_gate = 2π / |δ|  the motional mode returns to its
initial state and the spins acquire the entangling phase, realising the
Mølmer-Sørensen unitary  U_MS(π/2) = exp(−i π/4 σ_x⊗σ_x).

References
----------
Mintert & Wunderlich, Phys. Rev. Lett. **87**, 257904 (2001).
Ospelkaus et al., Phys. Rev. Lett. **101**, 090502 (2008).
Khromova et al., Phys. Rev. Lett. **108**, 220502 (2012).

Usage
-----
Command-line (text output only)::

    python magnetic_ms_gate.py

Interactive GUI::

    python magnetic_ms_gate.py --gui
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import qutip as qt
from matplotlib.gridspec import GridSpec
from matplotlib.widgets import Button, Slider

# ── physical constants ────────────────────────────────────────────────────────
MU_0 = 4.0 * np.pi * 1e-7   # vacuum permeability  (T m A⁻¹)
MU_B = 9.2740100783e-24      # Bohr magneton         (J T⁻¹)
HBAR = 1.0545718176e-34      # reduced Planck const  (J s)
G_S  = 2.002319476           # free-electron spin g-factor


# ══════════════════════════════════════════════════════════════════════════════
# Parameter dataclasses
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class WireMeanderGeometry:
    """Geometric and electrical parameters of the current-carrying wire meander.

    The meander consists of ``N_periods`` pairs of parallel wires carrying
    currents in alternating directions (±``current_amplitude``), separated by
    ``wire_pitch``.  The trapped ions sit at height ``trap_height`` above the
    wire plane.

    Parameters
    ----------
    wire_pitch :
        Centre-to-centre spacing between adjacent meander wires (m).
        Typical range: 10–200 µm.
    trap_height :
        Perpendicular distance from the wire plane to the ion trap (m).
        Typical range: 30–500 µm.
    current_amplitude :
        Peak current in each wire (A). Typical range: 0.05–2 A.
    N_periods :
        Number of meander periods (each period = one wire pair).
    wire_width :
        Width of each conductor (m).  Used for a first-order finite-width
        correction; set to 0 for infinitely thin wires.
    drive_frequency :
        Modulation frequency of the current (Hz).  Should be close to the
        ion secular frequency  ω_m / (2π).
    """

    wire_pitch        : float = 15e-6    # m
    trap_height       : float = 40e-6    # m
    current_amplitude : float = 1.0      # A
    N_periods         : int   = 8
    wire_width        : float = 2e-6     # m
    drive_frequency   : float = 1.0e6    # Hz

    # ── derived quantities ────────────────────────────────────────────────────

    def wire_positions(self) -> np.ndarray:
        """Return x-coordinates of meander wire centres (m).

        For *N_total = 2 × N_periods* wires, the positions are

            x_n = (n − (N_total−1)/2) × wire_pitch,   n = 0 … N_total−1

        with currents alternating +I, −I, +I, … .
        """
        N_total = 2 * self.N_periods
        n = np.arange(N_total, dtype=float)
        return (n - (N_total - 1) / 2.0) * self.wire_pitch

    def wire_currents(self) -> np.ndarray:
        """Return signed current (A) in each wire (+I, −I, +I, …)."""
        N_total = 2 * self.N_periods
        return self.current_amplitude * ((-1.0) ** np.arange(N_total))

    def field_gradient(self) -> float:
        r"""Compute the effective near-field gradient at the trap position (T m⁻¹).

        The meander acts as a spatially periodic near-field antenna.  Its
        dominant Fourier component has spatial wavevector  k₀ = π / p  (one
        full period = two adjacent wires of opposite polarity, spanning 2p).
        At height *d* above the wire plane the field amplitude decays as
        e^{−k₀ d}, giving an effective gradient

            |∂B/∂z| ≈ N_periods × (μ₀ I / p) × k₀ × e^{−k₀ d}
                     = N_periods × (μ₀ I π / p²) × e^{−π d / p}

        A first-order finite-width correction reduces the gradient by
        1 / (1 + (w/p)²) when *wire_width* > 0.

        Returns
        -------
        float
            Effective |∂B/∂z| in T m⁻¹.
        """
        p  = self.wire_pitch
        d  = self.trap_height
        I  = self.current_amplitude
        N  = self.N_periods
        w  = self.wire_width

        k0   = np.pi / p                           # fundamental wavevector
        grad = N * (MU_0 * I / p) * k0 * np.exp(-k0 * d)

        if w > 0:
            # finite-width correction: high spatial frequencies are suppressed
            grad /= 1.0 + (w / p) ** 2

        return grad

    def summary(self) -> str:
        """Return a human-readable summary of the geometry and derived gradient."""
        grad = self.field_gradient()
        lines = [
            "── Wire Meander Geometry ─────────────────",
            f"  Wire pitch           : {self.wire_pitch * 1e6:.1f}  µm",
            f"  Trap height          : {self.trap_height * 1e6:.1f}  µm",
            f"  Current amplitude    : {self.current_amplitude:.3f}  A",
            f"  Number of periods    : {self.N_periods}",
            f"  Wire width           : {self.wire_width * 1e6:.1f}  µm",
            f"  Drive frequency      : {self.drive_frequency / 1e6:.3f}  MHz",
            f"  ∂B/∂z (gradient)     : {grad:.3e}  T m⁻¹",
        ]
        return "\n".join(lines)


@dataclass
class IonParameters:
    """Physical parameters of the two trapped-ion qubits.

    Parameters
    ----------
    mass :
        Ion mass (kg). Default: ⁴⁰Ca⁺ ≈ 6.64 × 10⁻²⁶ kg.
    secular_frequency :
        Motional-mode (secular) angular frequency  ω_m  (rad s⁻¹).
    mean_phonon_number :
        Thermal occupation ⟨n⟩ of the motional mode at the start.
        0 = motional ground state.
    N_fock :
        Fock-state truncation for the motional mode Hilbert space.
        Must be large enough that truncation errors are negligible.
    """

    mass               : float = 6.6359e-26          # 40Ca+ (kg)
    secular_frequency  : float = 2.0 * np.pi * 1e6   # rad s⁻¹
    mean_phonon_number : float = 0.0
    N_fock             : int   = 10

    def zero_point_motion(self) -> float:
        """Return x_zpf = sqrt(ħ / 2 m ω_m) in metres."""
        return np.sqrt(HBAR / (2.0 * self.mass * self.secular_frequency))

    def lamb_dicke_parameter(self, gradient: float) -> float:
        r"""Compute the effective Lamb-Dicke-like parameter η.

            η = (g_s μ_B / ħ) × |∂B/∂z| × x_zpf / ω_m

        Parameters
        ----------
        gradient :
            Magnetic field gradient ∂B/∂z (T m⁻¹).

        Returns
        -------
        float
            Dimensionless η.
        """
        x_zpf    = self.zero_point_motion()
        coupling = G_S * MU_B * abs(gradient) * x_zpf / HBAR
        return coupling / self.secular_frequency

    def summary(self, gradient: float) -> str:
        """Return a human-readable summary of ion parameters and derived quantities."""
        eta   = self.lamb_dicke_parameter(gradient)
        x_zpf = self.zero_point_motion()
        lines = [
            "── Ion Parameters ────────────────────────",
            f"  Ion mass             : {self.mass:.3e}  kg",
            f"  Secular frequency    : {self.secular_frequency / (2*np.pi) / 1e6:.3f}  MHz",
            f"  Zero-point motion    : {x_zpf * 1e9:.3f}  nm",
            f"  Mean phonon number   : {self.mean_phonon_number:.2f}",
            f"  Fock basis size      : {self.N_fock}",
            f"  Lamb-Dicke param. η  : {eta:.4f}",
        ]
        return "\n".join(lines)


@dataclass
class GateParameters:
    """Parameters controlling the magnetic MS gate pulse.

    Parameters
    ----------
    detuning :
        Drive detuning from the motional sideband  δ = ω_drive − ω_m
        (rad s⁻¹).  The ideal gate time is  T_gate = 2π / |δ|.
    rabi_frequency :
        Carrier Rabi frequency  Ω_R  (rad s⁻¹), set by the oscillating
        magnetic field amplitude.
    N_steps :
        Number of time steps for the QuTiP ODE integrator output.
    include_decoherence :
        When True, add phenomenological T₁ / T₂ Lindblad operators.
    T1 :
        Longitudinal relaxation time T₁ (s).
    T2 :
        Total coherence time T₂ (s).  Includes T₁ contribution; must
        satisfy T₂ ≤ 2 T₁.
    """

    detuning            : float = 2.0 * np.pi * 10e3   # rad s⁻¹
    rabi_frequency      : float = 2.0 * np.pi * 435e3  # rad s⁻¹  (→ Ω_eff ≈ 0.434 δ)
    N_steps             : int   = 250
    include_decoherence : bool  = False
    T1                  : float = 1e-3   # s
    T2                  : float = 0.5e-3 # s

    def gate_time(self) -> float:
        """Return the ideal gate time T_gate = 2π / |δ| (s)."""
        return 2.0 * np.pi / abs(self.detuning)

    def summary(self, eta: float) -> str:
        """Return a human-readable summary of gate parameters."""
        T         = self.gate_time()
        Omega_eff = eta * self.rabi_frequency
        ratio     = abs(Omega_eff) / abs(self.detuning)
        lines = [
            "── Gate Parameters ───────────────────────",
            f"  Drive detuning   δ   : {self.detuning / (2*np.pi) / 1e3:.2f}  kHz",
            f"  Rabi frequency   Ω_R : {self.rabi_frequency / (2*np.pi) / 1e3:.2f}  kHz",
            f"  Eff. coupling  Ω_eff : {Omega_eff / (2*np.pi) / 1e3:.3f}  kHz",
            f"  Ω_eff / δ            : {ratio:.3f}  (gate cond. ≈ 0.434)",
            f"  Gate time   T_gate   : {T * 1e6:.2f}  µs",
            f"  Include decoherence  : {self.include_decoherence}",
        ]
        if self.include_decoherence:
            lines += [
                f"  T₁                   : {self.T1 * 1e3:.3f}  ms",
                f"  T₂                   : {self.T2 * 1e3:.3f}  ms",
            ]
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# Hamiltonian and collapse operators
# ══════════════════════════════════════════════════════════════════════════════

def _build_hamiltonian(
    ion     : IonParameters,
    gate    : GateParameters,
    gradient: float,
) -> Tuple[list, list, float, float]:
    """Construct the time-dependent QuTiP Hamiltonian.

    In the doubly-rotating frame the correct interaction-picture Hamiltonian
    for the bichromatic magnetic MS gate is

        H(t) = Ω_eff S_x (a e^{−iδt} + a† e^{+iδt})

    where  S_x = σ_x⁽¹⁾ ⊗ I + I ⊗ σ_x⁽²⁾  and the tensor-product ordering
    is  ion₁ ⊗ ion₂ ⊗ motion.  This drives a circular phase-space trajectory
    that closes at  T_gate = 2π/|δ|  and accumulates the entangling geometric
    phase.  The maximum-fidelity (F ≈ 1) gate condition is numerically

        Ω_eff ≈ 0.434 × |δ|

    The Hamiltonian is split into two separately time-varying terms for QuTiP:

        H = [Ω_eff S_x a] × e^{−iδt}  +  [Ω_eff S_x a†] × e^{+iδt}

    Note: QuTiP uses the ħ = 1 convention (H in rad s⁻¹).

    Parameters
    ----------
    ion, gate, gradient :
        Parameter objects and the magnetic field gradient (T m⁻¹).

    Returns
    -------
    H_list :
        QuTiP time-dependent Hamiltonian list suitable for sesolve/mesolve.
    dims :
        Hilbert-space dimensions ``[[2, 2, N_fock], [2, 2, N_fock]]``.
    delta :
        Gate detuning (rad s⁻¹).
    Omega_eff :
        Effective coupling rate (rad s⁻¹).
    """
    N         = ion.N_fock
    eta       = ion.lamb_dicke_parameter(gradient)
    Omega_eff = eta * gate.rabi_frequency
    delta     = gate.detuning

    # Spin and motional operators (QuTiP, ħ = 1)
    sx     = qt.sigmax()
    I_spin = qt.qeye(2)
    a      = qt.destroy(N)

    sx1 = qt.tensor(sx,     I_spin, qt.qeye(N))   # σ_x on ion 1
    sx2 = qt.tensor(I_spin, sx,     qt.qeye(N))   # σ_x on ion 2
    S_x = sx1 + sx2

    # H = Ω_eff S_x a * exp(-iδt)  +  Ω_eff S_x a† * exp(+iδt)
    H_a  = Omega_eff * S_x * qt.tensor(I_spin, I_spin, a)       # annihilation part
    H_ad = Omega_eff * S_x * qt.tensor(I_spin, I_spin, a.dag()) # creation part

    # Lambda coefficients capture the complex time dependence.
    # QuTiP 5 calls coeff(t, args) when args dict is provided to the solver.
    def _coeff_minus(t: float, args: dict) -> complex:
        return np.exp(-1j * args["delta"] * t)

    def _coeff_plus(t: float, args: dict) -> complex:
        return np.exp(1j * args["delta"] * t)

    H_list = [[H_a, _coeff_minus], [H_ad, _coeff_plus]]
    return H_list, H_a.dims, delta, Omega_eff


def _build_collapse_operators(
    ion : IonParameters,
    gate: GateParameters,
) -> List[qt.Qobj]:
    """Build Lindblad collapse operators for phenomenological decoherence.

    Adds amplitude-damping (T₁) and pure dephasing (T₂*) operators for
    each of the two spin qubits.

    Parameters
    ----------
    ion, gate :
        Parameter objects.

    Returns
    -------
    list of Qobj
    """
    N     = ion.N_fock
    I_s   = qt.qeye(2)
    I_m   = qt.qeye(N)
    sz    = qt.sigmaz()
    sm    = qt.sigmam()

    gamma1    = 1.0 / gate.T1
    gamma2    = 1.0 / gate.T2
    gamma_phi = max(gamma2 - gamma1 / 2.0, 0.0)   # pure-dephasing rate

    c_ops = []
    for ion_idx in range(2):
        if ion_idx == 0:
            op_decay = qt.tensor(sm,    I_s, I_m)
            op_deph  = qt.tensor(sz,    I_s, I_m)
        else:
            op_decay = qt.tensor(I_s, sm,    I_m)
            op_deph  = qt.tensor(I_s, sz,    I_m)

        c_ops.append(np.sqrt(gamma1)           * op_decay)
        c_ops.append(np.sqrt(gamma_phi / 2.0)  * op_deph)

    return c_ops


# ══════════════════════════════════════════════════════════════════════════════
# Ideal gate and fidelity
# ══════════════════════════════════════════════════════════════════════════════

def ideal_ms_gate(theta: float = np.pi / 2) -> qt.Qobj:
    r"""Return the ideal two-qubit Mølmer-Sørensen gate unitary.

        U_MS(θ) = exp(−i θ/2 · σ_x⊗σ_x)

    For maximum entanglement use θ = π/2.

    Parameters
    ----------
    theta :
        Rotation angle (rad).

    Returns
    -------
    qt.Qobj
        4×4 unitary matrix.
    """
    sxsx = qt.tensor(qt.sigmax(), qt.sigmax())
    return (-1j * theta / 2.0 * sxsx).expm()


def compute_state_fidelity(
    rho_final : qt.Qobj,
    psi_ideal : qt.Qobj,
) -> float:
    """Compute the state fidelity  F = ⟨ψ_ideal|ρ_final|ψ_ideal⟩.

    Parameters
    ----------
    rho_final :
        Final two-qubit spin density matrix.
    psi_ideal :
        Target pure state (ket).  Must have the same Hilbert-space
        dimensions as ``rho_final``.

    Returns
    -------
    float in [0, 1].
    """
    # Direct expectation value: F = ⟨ψ|ρ|ψ⟩
    # The multiplication ⟨ψ|ρ|ψ⟩ returns a complex scalar in QuTiP 5.
    val = psi_ideal.dag() * rho_final * psi_ideal
    # val may be a Qobj (1×1 matrix) or a bare complex number
    if isinstance(val, qt.Qobj):
        return float(np.real(val.tr()))
    return float(np.real(val))


# ══════════════════════════════════════════════════════════════════════════════
# Main simulation routine
# ══════════════════════════════════════════════════════════════════════════════

def run_simulation(
    wire     : WireMeanderGeometry,
    ion      : IonParameters,
    gate     : GateParameters,
) -> Dict[str, Any]:
    """Run the full magnetic MS gate simulation.

    Steps
    -----
    1. Compute ∂B/∂z from the wire geometry.
    2. Build the QuTiP time-dependent Hamiltonian.
    3. Prepare the initial state (pure ground state or thermal motional state).
    4. Integrate forward in time using ``sesolve`` (no decoherence) or
       ``mesolve`` (with decoherence / thermal state).
    5. Trace out the motional mode at each time step.
    6. Compute spin populations and gate fidelity at t = T_gate.

    Parameters
    ----------
    wire, ion, gate :
        Parameter objects.

    Returns
    -------
    dict with keys:
        ``"times"``        – 1-D array of times (s)
        ``"populations"``  – 2-D array [time, state_index] of spin pops
        ``"basis_labels"`` – list of 4 state label strings
        ``"fidelity"``     – gate fidelity (float in [0, 1])
        ``"gradient"``     – ∂B/∂z (T m⁻¹)
        ``"eta"``          – Lamb-Dicke parameter (dimensionless)
        ``"Omega_eff"``    – effective coupling rate (rad s⁻¹)
        ``"gate_time"``    – T_gate = 2π / |δ| (s)
    """
    gradient = wire.field_gradient()
    H_list, dims, delta, Omega_eff = _build_hamiltonian(ion, gate, gradient)

    T_gate = gate.gate_time()
    times  = np.linspace(0.0, T_gate, gate.N_steps)

    # ── initial state ─────────────────────────────────────────────────────────
    n_bar    = ion.mean_phonon_number
    N        = ion.N_fock
    psi_spin = qt.tensor(qt.basis(2, 0), qt.basis(2, 0))   # |↑↑⟩

    use_dm = (n_bar > 0.0) or gate.include_decoherence

    if not use_dm:
        psi0       = qt.tensor(psi_spin, qt.basis(N, 0))
        c_ops: list = []
    else:
        rho_motion = qt.thermal_dm(N, n_bar)
        psi0       = qt.tensor(qt.ket2dm(psi_spin), rho_motion)
        c_ops      = _build_collapse_operators(ion, gate) if gate.include_decoherence else []

    # ── time evolution ────────────────────────────────────────────────────────
    args    = {"delta": delta}
    options = {"nsteps": 50_000, "rtol": 1e-8, "atol": 1e-10}

    if use_dm or c_ops:
        result = qt.mesolve(H_list, psi0, times,
                            c_ops=c_ops, args=args, options=options)
    else:
        result = qt.sesolve(H_list, psi0, times,
                            args=args, options=options)

    states = result.states

    # ── trace out motion, compute spin density matrices ───────────────────────
    def _spin_dm(state: qt.Qobj) -> qt.Qobj:
        dm = qt.ket2dm(state) if state.type == "ket" else state
        return dm.ptrace([0, 1])   # keep ions 0 and 1, trace out motion (2)

    spin_states = [_spin_dm(s) for s in states]

    # ── populations of the 4 computational spin basis states ─────────────────
    basis_labels = ["|↑↑⟩", "|↑↓⟩", "|↓↑⟩", "|↓↓⟩"]
    pops = np.zeros((len(times), 4))
    for i, rho_s in enumerate(spin_states):
        # Diagonal of the 4×4 spin density matrix gives the populations in
        # the {|↑↑⟩, |↑↓⟩, |↓↑⟩, |↓↓⟩} basis (ptrace returns dims [[2,2],[2,2]])
        pops[i] = np.diag(rho_s.full()).real

    # ── gate fidelity (state fidelity for |↑↑⟩ input) ────────────────────────
    # Ideal MS gate output for |↑↑⟩: (|↑↑⟩ − i|↓↓⟩) / √2
    # Build as tensor product so dims match the ptrace output [[2,2],[2,2]]
    psi_ideal = (
        qt.tensor(qt.basis(2, 0), qt.basis(2, 0))
        - 1j * qt.tensor(qt.basis(2, 1), qt.basis(2, 1))
    ) / np.sqrt(2.0)
    fidelity  = compute_state_fidelity(spin_states[-1], psi_ideal)

    return {
        "times"       : times,
        "populations" : pops,
        "basis_labels": basis_labels,
        "fidelity"    : fidelity,
        "gradient"    : gradient,
        "eta"         : ion.lamb_dicke_parameter(gradient),
        "Omega_eff"   : Omega_eff,
        "gate_time"   : T_gate,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Parameter sweeps
# ══════════════════════════════════════════════════════════════════════════════

def sweep_parameter(
    param_name    : str,
    param_values  : np.ndarray,
    wire          : WireMeanderGeometry,
    ion           : IonParameters,
    gate          : GateParameters,
) -> np.ndarray:
    """Sweep one numeric parameter and collect gate fidelity values.

    Parameters
    ----------
    param_name :
        Dot-delimited path to the attribute to vary, prefixed by the
        object name: ``"wire.wire_pitch"``, ``"ion.secular_frequency"``,
        ``"gate.detuning"``, ``"gate.rabi_frequency"``, etc.
    param_values :
        1-D array of values to set for that attribute.
    wire, ion, gate :
        Baseline parameter objects (not mutated).

    Returns
    -------
    np.ndarray
        Gate fidelity at each parameter value.
    """
    import copy

    obj_name, attr_name = param_name.split(".", 1)
    obj_map = {"wire": wire, "ion": ion, "gate": gate}
    if obj_name not in obj_map:
        raise ValueError(f"Unknown object '{obj_name}'; use 'wire', 'ion', or 'gate'.")

    fidelities = np.empty(len(param_values))
    for k, val in enumerate(param_values):
        w_k = copy.deepcopy(wire)
        i_k = copy.deepcopy(ion)
        g_k = copy.deepcopy(gate)
        obj = {"wire": w_k, "ion": i_k, "gate": g_k}[obj_name]
        setattr(obj, attr_name, val)
        try:
            res = run_simulation(w_k, i_k, g_k)
            fidelities[k] = res["fidelity"]
        except Exception:
            fidelities[k] = np.nan

    return fidelities


# ══════════════════════════════════════════════════════════════════════════════
# Matplotlib-widgets interactive GUI
# ══════════════════════════════════════════════════════════════════════════════

class MagneticMSGateGUI:
    """Interactive matplotlib GUI for the Magnetic MS Gate simulation.

    Layout (3 columns, 2 rows of plots + slider panel)
    ---------------------------------------------------
    [Top-left]    Spin-state populations vs time
    [Top-centre]  Phase-space displacement |α(t)|
    [Top-right]   Fidelity vs gate detuning sweep
    [Bot-left]    Fidelity vs Rabi frequency sweep
    [Bot-centre]  Fidelity vs wire current sweep
    [Bot-right]   Summary text

    Sliders control all key parameters; "Run" triggers a fresh simulation.
    """

    # Slider definitions: (label, min, max, default, scale, attr_path)
    _SLIDERS: List[Tuple[str, float, float, float, float, str]] = [
        # label                min     max     default   scale  attr_path
        ("Wire pitch (µm)",     5,      100,    15,       1e-6,  "wire.wire_pitch"),
        ("Trap height (µm)",    10,     200,    40,       1e-6,  "wire.trap_height"),
        ("Current (A)",         0.1,    3.0,    1.0,      1.0,   "wire.current_amplitude"),
        ("N periods",           1,      16,     8,        1.0,   "wire.N_periods"),
        ("ω_m / 2π (MHz)",      0.2,    5.0,    1.0,      2e6*np.pi, "ion.secular_frequency"),
        ("⟨n⟩ thermal",         0.0,    5.0,    0.0,      1.0,   "ion.mean_phonon_number"),
        ("Detuning δ (kHz)",    1.0,    100.0,  10.0,     2e3*np.pi, "gate.detuning"),
        ("Rabi Ω_R (kHz)",      50.0,   2000.0, 435.0,    2e3*np.pi, "gate.rabi_frequency"),
    ]

    def __init__(self) -> None:
        self._wire = WireMeanderGeometry()
        self._ion  = IonParameters()
        self._gate = GateParameters()

        self._fig = plt.figure(figsize=(18, 10))
        self._fig.canvas.manager.set_window_title(
            "Magnetic Mølmer-Sørensen Gate — QuTiP Simulation"
        )
        self._build_layout()
        self._build_sliders()
        self._run_simulation()

    # ── layout ────────────────────────────────────────────────────────────────

    def _build_layout(self) -> None:
        gs = GridSpec(
            3, 3,
            figure=self._fig,
            top=0.94, bottom=0.38, hspace=0.45, wspace=0.35,
            left=0.06, right=0.97,
        )
        self._ax_pops   = self._fig.add_subplot(gs[0, 0])
        self._ax_phase  = self._fig.add_subplot(gs[0, 1])
        self._ax_fid_d  = self._fig.add_subplot(gs[0, 2])
        self._ax_fid_r  = self._fig.add_subplot(gs[1, 0])
        self._ax_fid_I  = self._fig.add_subplot(gs[1, 1])
        self._ax_text   = self._fig.add_subplot(gs[1, 2])
        self._ax_text.axis("off")

    def _build_sliders(self) -> None:
        n   = len(self._SLIDERS)
        col = 4                               # sliders per row
        row = (n + col - 1) // col
        W, H = 0.19, 0.025
        x0, y0 = 0.06, 0.30
        dx, dy = 0.245, 0.045

        self._sliders: List[Slider] = []
        for k, (label, vmin, vmax, vdef, _, _attr) in enumerate(self._SLIDERS):
            r, c = divmod(k, col)
            ax   = self._fig.add_axes([x0 + c * dx, y0 - r * dy, W, H])
            sl   = Slider(ax, label, vmin, vmax, valinit=vdef, color="#4c92c3")
            sl.label.set_fontsize(8)
            sl.valtext.set_fontsize(8)
            self._sliders.append(sl)

        # Run button
        btn_ax = self._fig.add_axes([0.44, 0.30 - (row - 1) * dy - 0.06, 0.12, 0.035])
        self._btn = Button(btn_ax, "▶  Run Simulation", color="#e8f4e8",
                           hovercolor="#b4ddb4")
        self._btn.on_clicked(self._on_run)

    # ── callbacks ─────────────────────────────────────────────────────────────

    def _on_run(self, _event: Any = None) -> None:
        self._read_sliders()
        self._run_simulation()

    def _read_sliders(self) -> None:
        import copy
        for sl, (_label, _min, _max, _def, scale, attr_path) in zip(
            self._sliders, self._SLIDERS
        ):
            obj_name, attr_name = attr_path.split(".", 1)
            obj = {"wire": self._wire, "ion": self._ion, "gate": self._gate}[obj_name]
            raw = sl.val
            # N_periods must be integer
            if attr_name == "N_periods":
                raw = int(round(raw))
            setattr(obj, attr_name, raw * scale)

    # ── simulation & plotting ─────────────────────────────────────────────────

    def _run_simulation(self) -> None:
        try:
            res = run_simulation(self._wire, self._ion, self._gate)
        except Exception as exc:
            for ax in (self._ax_pops, self._ax_phase, self._ax_fid_d,
                       self._ax_fid_r, self._ax_fid_I):
                ax.clear()
                ax.text(0.5, 0.5, f"Simulation error:\n{exc}",
                        ha="center", va="center", transform=ax.transAxes,
                        color="red", fontsize=8)
            self._fig.canvas.draw_idle()
            return

        self._plot_populations(res)
        self._plot_phase_space(res)
        self._plot_sweeps(res)
        self._plot_summary(res)
        self._fig.canvas.draw_idle()

    def _plot_populations(self, res: dict) -> None:
        ax     = self._ax_pops
        times  = res["times"] * 1e6          # µs
        pops   = res["populations"]
        labels = res["basis_labels"]
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

        ax.clear()
        for j, (lbl, col) in enumerate(zip(labels, colors)):
            ax.plot(times, pops[:, j], label=lbl, color=col)
        ax.set_xlabel("Time (µs)", fontsize=8)
        ax.set_ylabel("Population", fontsize=8)
        ax.set_title("Spin populations vs time", fontsize=9)
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=7, ncol=2)
        ax.tick_params(labelsize=7)
        ax.text(0.97, 0.97,
                f"F = {res['fidelity']:.4f}",
                ha="right", va="top",
                transform=ax.transAxes, fontsize=9,
                bbox=dict(boxstyle="round,pad=0.2", fc="wheat", alpha=0.8))

    def _plot_phase_space(self, res: dict) -> None:
        """Plot the coherent phase-space displacement α(t) traced by the motional mode.

        For the interaction-picture Hamiltonian H(t) = Ω_eff S_x (a e^{−iδt} + a† e^{+iδt}),
        the displacement of the motional coherent state is

            α(t) = Ω_eff / δ × (1 − e^{iδt})

        which traces a circle in phase space that closes at t = T_gate = 2π/δ.
        The enclosed area determines the entangling geometric phase.
        """
        ax        = self._ax_phase
        delta     = abs(self._gate.detuning)
        Omega_eff = res["Omega_eff"]
        times     = res["times"]

        # Phase-space displacement: α(t) = Ω_eff/δ × (1 - exp(iδt))
        alpha = Omega_eff / delta * (1.0 - np.exp(1j * delta * times))
        ax.clear()
        ax.plot(alpha.real, alpha.imag, color="#8b2be2", lw=1.5)
        ax.plot(alpha.real[0],  alpha.imag[0],  "go", ms=6, label="start")
        ax.plot(alpha.real[-1], alpha.imag[-1], "r*", ms=8, label="end")
        ax.set_xlabel("Re(α)", fontsize=8)
        ax.set_ylabel("Im(α)", fontsize=8)
        ax.set_title("Phase-space loop α(t)", fontsize=9)
        ax.set_aspect("equal", "box")
        ax.legend(fontsize=7)
        ax.tick_params(labelsize=7)
        r_max = max(abs(alpha.real).max(), abs(alpha.imag).max()) * 1.3 + 1e-10
        ax.set_xlim(-r_max, r_max)
        ax.set_ylim(-r_max, r_max)

    def _plot_sweeps(self, res: dict) -> None:
        # Fidelity vs detuning
        d_vals = np.linspace(2e3, 80e3, 30) * 2.0 * np.pi   # rad/s
        f_d    = sweep_parameter("gate.detuning", d_vals,
                                 self._wire, self._ion, self._gate)
        ax = self._ax_fid_d
        ax.clear()
        ax.plot(d_vals / (2e3 * np.pi), f_d, color="#1f77b4")
        ax.axvline(abs(self._gate.detuning) / (2e3 * np.pi),
                   color="red", ls="--", lw=0.8)
        ax.set_xlabel("Detuning δ (kHz)", fontsize=8)
        ax.set_ylabel("Fidelity", fontsize=8)
        ax.set_title("Fidelity vs detuning", fontsize=9)
        ax.set_ylim(-0.05, 1.05)
        ax.tick_params(labelsize=7)

        # Fidelity vs Rabi frequency
        r_vals = np.linspace(50e3, 2000e3, 30) * 2.0 * np.pi
        f_r    = sweep_parameter("gate.rabi_frequency", r_vals,
                                 self._wire, self._ion, self._gate)
        ax = self._ax_fid_r
        ax.clear()
        ax.plot(r_vals / (2e3 * np.pi), f_r, color="#ff7f0e")
        ax.axvline(abs(self._gate.rabi_frequency) / (2e3 * np.pi),
                   color="red", ls="--", lw=0.8)
        ax.set_xlabel("Rabi Ω_R (kHz)", fontsize=8)
        ax.set_ylabel("Fidelity", fontsize=8)
        ax.set_title("Fidelity vs Rabi frequency", fontsize=9)
        ax.set_ylim(-0.05, 1.05)
        ax.tick_params(labelsize=7)

        # Fidelity vs wire current
        I_vals = np.linspace(0.1, 3.0, 20)
        f_I    = sweep_parameter("wire.current_amplitude", I_vals,
                                 self._wire, self._ion, self._gate)
        ax = self._ax_fid_I
        ax.clear()
        ax.plot(I_vals, f_I, color="#2ca02c")
        ax.axvline(self._wire.current_amplitude, color="red", ls="--", lw=0.8)
        ax.set_xlabel("Wire current (A)", fontsize=8)
        ax.set_ylabel("Fidelity", fontsize=8)
        ax.set_title("Fidelity vs wire current", fontsize=9)
        ax.set_ylim(-0.05, 1.05)
        ax.tick_params(labelsize=7)

    def _plot_summary(self, res: dict) -> None:
        ax = self._ax_text
        ax.clear()
        ax.axis("off")
        grad = res["gradient"]
        eta  = res["eta"]
        text = (
            f"∂B/∂z  = {grad:.3e} T m⁻¹\n"
            f"η      = {eta:.4f}\n"
            f"Ω_eff  = {res['Omega_eff'] / (2e3*np.pi):.3f} kHz\n"
            f"T_gate = {res['gate_time']*1e6:.2f} µs\n"
            f"Fidelity = {res['fidelity']:.4f}"
        )
        ax.text(0.05, 0.55, text, transform=ax.transAxes, fontsize=9,
                va="center", family="monospace",
                bbox=dict(boxstyle="round,pad=0.5", fc="#f0f4ff", alpha=0.9))
        ax.set_title("Derived quantities", fontsize=9)

    # ── public API ─────────────────────────────────────────────────────────────

    def show(self) -> None:
        """Display the GUI window (blocks until closed)."""
        plt.suptitle(
            "Magnetic Mølmer-Sørensen Gate  –  QuTiP Simulation",
            fontsize=12, fontweight="bold",
        )
        plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# Command-line entry-point
# ══════════════════════════════════════════════════════════════════════════════

def _print_banner() -> None:
    print("=" * 60)
    print("  Magnetic Mølmer-Sørensen Gate  —  QuTiP Simulation")
    print("=" * 60)


def main(argv: Optional[List[str]] = None) -> None:
    """Run the simulation from the command line.

    Flags
    -----
    ``--gui``       Open the interactive matplotlib GUI.
    ``--no-plot``   Skip all plotting (useful for scripting / CI).
    """
    parser = argparse.ArgumentParser(
        description="Magnetic MS gate simulation using QuTiP."
    )
    parser.add_argument("--gui",     action="store_true",
                        help="Launch the interactive GUI")
    parser.add_argument("--no-plot", action="store_true",
                        help="Skip all matplotlib plots")
    args = parser.parse_args(argv)

    _print_banner()

    if args.gui:
        matplotlib.use("TkAgg" if "tkinter" in sys.modules else "Qt5Agg")
        try:
            gui = MagneticMSGateGUI()
            gui.show()
        except Exception:
            # Fall back to non-interactive Agg backend
            matplotlib.use("Agg")
            gui = MagneticMSGateGUI()
            gui._fig.savefig("magnetic_ms_gate_gui.png", dpi=150,
                             bbox_inches="tight")
            print("GUI saved to magnetic_ms_gate_gui.png")
        return

    # ── default: single simulation with printed summary ──────────────────────
    wire = WireMeanderGeometry()
    ion  = IonParameters()
    gate = GateParameters()

    print(wire.summary())
    print()
    grad = wire.field_gradient()
    print(ion.summary(grad))
    print()
    eta = ion.lamb_dicke_parameter(grad)
    print(gate.summary(eta))
    print()

    print("Running simulation …", flush=True)
    res = run_simulation(wire, ion, gate)

    print(f"\n  Gate time    : {res['gate_time'] * 1e6:.2f} µs")
    print(f"  η (Lamb-Dicke): {res['eta']:.4f}")
    print(f"  Gradient     : {res['gradient']:.3e} T m⁻¹")
    print(f"  Ω_eff        : {res['Omega_eff'] / (2e3*np.pi):.3f} kHz")
    print(f"  Gate fidelity: {res['fidelity']:.4f}")

    print("\nFinal spin-state populations:")
    pops_final = res["populations"][-1]
    for lbl, p in zip(res["basis_labels"], pops_final):
        print(f"  {lbl} : {p:.4f}")

    if not args.no_plot:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        fig.suptitle("Magnetic MS Gate – QuTiP Simulation", fontsize=12)

        # Populations
        times  = res["times"] * 1e6
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
        for j, (lbl, col) in enumerate(zip(res["basis_labels"], colors)):
            axes[0].plot(times, res["populations"][:, j], label=lbl, color=col)
        axes[0].set_xlabel("Time (µs)")
        axes[0].set_ylabel("Population")
        axes[0].set_title("Spin populations")
        axes[0].legend(fontsize=8)
        axes[0].set_ylim(-0.05, 1.05)

        # Phase-space loop: α(t) = Ω_eff/δ × (1 - exp(iδt))
        delta     = abs(gate.detuning)
        Omega_eff = res["Omega_eff"]
        ts        = res["times"]
        alpha     = Omega_eff / delta * (1.0 - np.exp(1j * delta * ts))
        axes[1].plot(alpha.real, alpha.imag, color="#8b2be2")
        axes[1].plot(alpha.real[0],  alpha.imag[0],  "go", ms=8, label="start")
        axes[1].plot(alpha.real[-1], alpha.imag[-1], "r*", ms=10, label="end")
        axes[1].set_xlabel("Re(α)")
        axes[1].set_ylabel("Im(α)")
        axes[1].set_title("Phase-space loop")
        axes[1].set_aspect("equal", "box")
        axes[1].legend()

        # Fidelity vs detuning
        d_vals = np.linspace(2e3, 80e3, 25) * 2.0 * np.pi
        f_d    = sweep_parameter("gate.detuning", d_vals, wire, ion, gate)
        axes[2].plot(d_vals / (2e3 * np.pi), f_d, color="#1f77b4")
        axes[2].axvline(abs(gate.detuning) / (2e3 * np.pi),
                        color="red", ls="--", label="current δ")
        axes[2].set_xlabel("Detuning δ (kHz)")
        axes[2].set_ylabel("Fidelity")
        axes[2].set_title("Fidelity vs detuning")
        axes[2].set_ylim(-0.05, 1.05)
        axes[2].legend()

        fig.tight_layout()
        plt.savefig("magnetic_ms_gate.png", dpi=150, bbox_inches="tight")
        print("\nPlot saved to magnetic_ms_gate.png")
        plt.show()


if __name__ == "__main__":
    main()
