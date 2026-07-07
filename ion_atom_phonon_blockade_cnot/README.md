# Ion-Atom Phonon Blockade CNOT Colab Project

This project is a compact Colab-friendly reproduction of the simulations in:

```text
arXiv:2602.19222v2
Ion-atom two-qubit quantum gate based on phonon blockade
Subhra Mudli and Bimalendu Deb
```

It reproduces the main numerical ingredients:

- shifted ion phonon frequency and equilibrium displacement versus ion-atom distance;
- three-pulse atom-ion-phonon amplitude traces for the phonon blockade CNOT protocol;
- average CNOT fidelity versus atomic Rabi frequency.

## Files

- `ion_atom_gate.py`: simulation code and plotting entry point.
- `Ion_Atom_Phonon_Blockade_Colab.ipynb`: upload-and-run Colab notebook.
- `requirements-colab.txt`: Colab dependencies.

## Local Run

```bash
python3 ion_atom_phonon_blockade_colab/ion_atom_gate.py
```

Outputs are saved under:

```text
ion_atom_phonon_blockade_colab/outputs/
```

Main figures:

```text
fig2_shifted_trap.png
fig3_amplitude_traces.png
fig4_fidelity_scan.png
```

## Colab Bundle

From the parent `Code` directory:

```bash
zip -qr ion_atom_phonon_blockade_colab.zip ion_atom_phonon_blockade_colab -x "ion_atom_phonon_blockade_colab/outputs/*"
```

Open `Ion_Atom_Phonon_Blockade_Colab.ipynb` in Colab, upload the zip, and run the cells.

## Notes

The simulation uses angular frequencies internally. The default constants are chosen from the paper's
87Rb-9Be+ example:

- unperturbed ion trap frequency: `2pi * 11.2 MHz`;
- ion-atom distance: `2.57 um`;
- Rydberg state: `87Rb n=90`;
- `C4 = 5.07e10 * 160 a.u.`;
- Lamb-Dicke parameter: `eta = 0.1`;
- ion Rabi frequency: `2pi * 1 MHz`;
- default atom Rabi frequency: `2pi * 1 GHz`.
- effective Rydberg-pulse phonon leakage scale: `0.46`, calibrated so the compact model gives about
  `90%` fidelity near `Omega_a / 2pi = 1 GHz`, as reported in the paper.

The model keeps a small phonon Hilbert space and includes the paper's two central mechanisms: Rydberg-state
phonon perturbations during atomic pulses, and detuned red-sideband dynamics during the ionic pulse.
Set `rydberg_phonon_leakage_scale=0` in `PhysicalParams` for the ideal fast-Rydberg-pulse limit.
