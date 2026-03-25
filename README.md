# QuTiP Simulations

A Python package for quantum physics simulations built on top of [QuTiP](https://qutip.org/).

## Features

- **QubitSimulation** – single-qubit dynamics (Rabi oscillations, spontaneous emission)
- **HarmonicOscillatorSimulation** – driven and damped quantum harmonic oscillator
- **LindbladSimulation** – general Lindblad master-equation solver for open quantum systems

## Installation

```bash
pip install -r requirements.txt
pip install -e .
```

## Quick Start

### Qubit: Rabi oscillations

```python
from qutip_simulations import QubitSimulation

sim = QubitSimulation(omega=1.0, drive_amplitude=0.5, drive_frequency=1.0)
result = sim.run(tmax=20.0, steps=500)
# result.times  – time array
# result.expect – [<σ_x>, <σ_y>, <σ_z>]
```

### Harmonic oscillator: cavity decay

```python
import qutip as qt
from qutip_simulations import HarmonicOscillatorSimulation

fock3 = qt.basis(20, 3)
sim = HarmonicOscillatorSimulation(omega=1.0, n_fock=20, kappa=0.2, initial_state=fock3)
result = sim.run(tmax=15.0, steps=300)
# result.expect[0] – mean photon number ⟨n⟩
```

### Open quantum system: Lindblad master equation

```python
import numpy as np
import qutip as qt
from qutip_simulations import LindbladSimulation

H = 0.5 * qt.sigmaz()
c_ops = [np.sqrt(0.5) * qt.sigmam()]
rho0 = qt.ket2dm(qt.basis(2, 0))   # excited state (basis(2, 0) in QuTiP)
sim = LindbladSimulation(H, c_ops, rho0)
result = sim.run(tmax=10.0, steps=100, observables=[qt.sigmaz()])
ss = sim.steady_state()            # steady-state density matrix
```

## Running the Tests

```bash
pip install -r requirements-dev.txt
pytest
```

## Examples

```bash
python examples/qubit_dynamics.py
python examples/cavity_qed.py
```

## License

MIT
