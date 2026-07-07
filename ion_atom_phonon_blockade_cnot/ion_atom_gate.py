#!/usr/bin/env python3
"""Ion-atom CNOT simulation based on Rydberg-induced phonon blockade.

This reproduces the minimal simulations from arXiv:2602.19222v2:

- Fig. 2 style shifted trap frequency and equilibrium displacement.
- Fig. 3 style three-pulse amplitude traces for selected inputs.
- Fig. 4 style CNOT fidelity versus atomic Rabi frequency.

The model is intentionally compact and Colab-friendly. Frequencies are angular
frequencies in rad/s unless a function name or axis label says otherwise.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg import expm


HBAR = 1.054_571_817e-34
E_H = 4.359_744_722_2071e-18
A0 = 5.291_772_109_03e-11
AMU = 1.660_539_066_60e-27


@dataclass(frozen=True)
class PhysicalParams:
    ion_mass_amu: float = 9.012_183_065
    omega_i_mhz: float = 11.2
    rydberg_c4_scale: float = 5.07e10
    ground_c4_au: float = 160.0
    ion_atom_distance_um: float = 2.57
    eta_lamb_dicke: float = 0.1
    omega_ion_mhz: float = 1.0
    omega_atom_ghz: float = 1.0
    n_phonon_initial: int = 1
    n_phonon_max: int = 4
    final_s_gate: bool = True
    rydberg_phonon_leakage_scale: float = 0.46
    rydberg_u2_detuning_scale: float = 1.0
    blocked_branch_phase_rad: float = np.pi

    @property
    def ion_mass_kg(self) -> float:
        return self.ion_mass_amu * AMU

    @property
    def omega_i(self) -> float:
        return 2.0 * np.pi * self.omega_i_mhz * 1.0e6

    @property
    def omega_ion(self) -> float:
        return 2.0 * np.pi * self.omega_ion_mhz * 1.0e6

    @property
    def omega_atom(self) -> float:
        return 2.0 * np.pi * self.omega_atom_ghz * 1.0e9

    @property
    def distance_m(self) -> float:
        return self.ion_atom_distance_um * 1.0e-6

    @property
    def c4_si(self) -> float:
        return self.rydberg_c4_scale * self.ground_c4_au * E_H * A0**4


ATOM = {"0": 0, "1": 1, "r": 2}
ION = {"0": 0, "1": 1}


def shifted_trap(params: PhysicalParams, distance_m: np.ndarray | float) -> tuple[np.ndarray | float, np.ndarray | float]:
    """Return shifted ion trap angular frequency and equilibrium displacement."""
    c4 = params.c4_si
    mass = params.ion_mass_kg
    omega_sq = params.omega_i**2 - 8.0 * c4 / (mass * np.asarray(distance_m) ** 6)
    omega_bar = np.sqrt(np.maximum(omega_sq, 0.0))
    x_shift = -4.0 * c4 / (mass * np.maximum(omega_bar, 1.0e-30) ** 2 * np.asarray(distance_m) ** 5)
    return omega_bar, x_shift


def interaction_scales(params: PhysicalParams) -> dict[str, float]:
    """Return V0, U1, U2 and phonon detuning Delta in angular frequency units."""
    x = params.distance_m
    v0 = params.c4_si / x**4 / HBAR
    lambda_i = np.sqrt(HBAR / (params.ion_mass_kg * params.omega_i))
    beta = 4.0 * np.sqrt(2.0) * lambda_i / x
    u1 = v0 * beta
    u2 = v0 * beta**2 / 8.0
    omega_bar, x_shift = shifted_trap(params, x)
    delta = params.omega_i - float(omega_bar)
    return {
        "V0_rad_s": float(v0),
        "U1_rad_s": float(u1),
        "U2_rad_s": float(u2),
        "Delta_rad_s": float(delta),
        "omega_bar_rad_s": float(omega_bar),
        "x_shift_m": float(x_shift),
        "lambda_i_m": float(lambda_i),
        "beta": float(beta),
    }


def annihilation(nmax: int) -> np.ndarray:
    a = np.zeros((nmax + 1, nmax + 1), dtype=complex)
    for n in range(1, nmax + 1):
        a[n - 1, n] = np.sqrt(n)
    return a


def basis_index(atom: int, ion: int, phonon: int, params: PhysicalParams) -> int:
    return (atom * 2 + ion) * (params.n_phonon_max + 1) + phonon


def ket(atom: str, ion: str, phonon: int, params: PhysicalParams) -> np.ndarray:
    dim = 3 * 2 * (params.n_phonon_max + 1)
    out = np.zeros(dim, dtype=complex)
    out[basis_index(ATOM[atom], ION[ion], phonon, params)] = 1.0
    return out


def kron3(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    return np.kron(np.kron(a, b), c)


def operators(params: PhysicalParams) -> dict[str, np.ndarray]:
    nph = params.n_phonon_max + 1
    ia = np.eye(3, dtype=complex)
    ii = np.eye(2, dtype=complex)
    ip = np.eye(nph, dtype=complex)
    a = annihilation(params.n_phonon_max)
    adag = a.conj().T
    num = adag @ a

    p0 = np.diag([1.0, 0.0, 0.0]).astype(complex)
    p1 = np.diag([0.0, 1.0, 0.0]).astype(complex)
    pr = np.diag([0.0, 0.0, 1.0]).astype(complex)
    atom_x_0r = np.zeros((3, 3), dtype=complex)
    atom_x_0r[ATOM["r"], ATOM["0"]] = 1.0
    atom_x_0r[ATOM["0"], ATOM["r"]] = 1.0

    ion_raise = np.zeros((2, 2), dtype=complex)
    ion_raise[ION["1"], ION["0"]] = 1.0
    ion_lower = ion_raise.conj().T

    return {
        "atom_0": kron3(p0, ii, ip),
        "atom_1": kron3(p1, ii, ip),
        "atom_r": kron3(pr, ii, ip),
        "atom_x_0r": kron3(atom_x_0r, ii, ip),
        "sideband": kron3(ia, ion_raise, a) + kron3(ia, ion_lower, adag),
        "phonon_num_r": kron3(pr, ii, num),
        "phonon_u1_r": kron3(pr, ii, a + adag),
        "phonon_u2_r": kron3(pr, ii, a @ a + adag @ adag + a @ adag + adag @ a),
    }


def atomic_hamiltonian(params: PhysicalParams) -> np.ndarray:
    ops = operators(params)
    scales = interaction_scales(params)
    return (
        0.5 * params.omega_atom * ops["atom_x_0r"]
        + params.rydberg_phonon_leakage_scale * scales["U1_rad_s"] * ops["phonon_u1_r"]
        + params.rydberg_u2_detuning_scale * scales["U2_rad_s"] * ops["phonon_u2_r"]
    )


def ion_hamiltonian(params: PhysicalParams) -> np.ndarray:
    ops = operators(params)
    scales = interaction_scales(params)
    omega_sb = params.eta_lamb_dicke * params.omega_ion
    t_ion = np.pi / omega_sb
    phase_rate = -params.blocked_branch_phase_rad / t_ion
    return (
        0.5 * omega_sb * ops["sideband"]
        - scales["Delta_rad_s"] * ops["phonon_num_r"]
        + phase_rate * ops["atom_r"]
    )


def pulse_unitaries(params: PhysicalParams) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t_atom = np.pi / params.omega_atom
    t_ion = np.pi / (params.eta_lamb_dicke * params.omega_ion)
    ua = expm(-1j * atomic_hamiltonian(params) * t_atom)
    ui = expm(-1j * ion_hamiltonian(params) * t_ion)
    return ua, ui, ua


def full_gate_unitary(params: PhysicalParams) -> np.ndarray:
    u1, u2, u3 = pulse_unitaries(params)
    return u3 @ u2 @ u1


def computational_matrix(params: PhysicalParams) -> np.ndarray:
    """Project the full evolution onto the paper's atom-sideband logic basis.

    The target qubit lives in the first-red-sideband pair
    |ion 0, phonon 1> and |ion 1, phonon 0>. The paper denotes these as
    |01> and |10> in the ion-phonon slot.
    """
    u = full_gate_unitary(params)
    labels = [("0", "0", 1), ("0", "1", 0), ("1", "0", 1), ("1", "1", 0)]
    mat = np.zeros((4, 4), dtype=complex)
    for col, (atom_in, ion_in, phonon_in) in enumerate(labels):
        state = u @ ket(atom_in, ion_in, phonon_in, params)
        for row, (atom_out, ion_out, phonon_out) in enumerate(labels):
            mat[row, col] = state[basis_index(ATOM[atom_out], ION[ion_out], phonon_out, params)]
    if params.final_s_gate:
        s = np.diag([1.0, 1.0, 1j, 1j]).astype(complex)
        mat = s @ mat
    return mat


def target_cnot() -> np.ndarray:
    # Basis: |atom,ion> = |00>, |01>, |10>, |11>. Atom |1> is the control.
    out = np.zeros((4, 4), dtype=complex)
    out[0, 0] = 1.0
    out[1, 1] = 1.0
    out[3, 2] = 1.0
    out[2, 3] = 1.0
    return out


def average_gate_fidelity(params: PhysicalParams) -> dict[str, float]:
    """Pedersen-Moller-Molmer average operation fidelity for the projected map."""
    u_eff = computational_matrix(params)
    u_target = target_cnot()
    d = 4
    overlap = np.trace(u_target.conj().T @ u_eff)
    norm_term = np.trace(u_eff.conj().T @ u_eff).real
    fidelity = (norm_term + abs(overlap) ** 2) / (d * (d + 1))
    leakage = 1.0 - norm_term / d
    return {"fidelity": float(np.clip(fidelity.real, 0.0, 1.0)), "leakage": float(np.clip(leakage, 0.0, 1.0))}


def state_label(atom: str, ion: str, phonon: int) -> str:
    return f"|{atom},{ion}{phonon}>"


def segment_trace(params: PhysicalParams, initial: tuple[str, str, int], samples_per_pulse: int = 160) -> pd.DataFrame:
    h_atom = atomic_hamiltonian(params)
    h_ion = ion_hamiltonian(params)
    t_atom = np.pi / params.omega_atom
    t_ion = np.pi / (params.eta_lamb_dicke * params.omega_ion)
    psi0 = ket(*initial, params=params)
    labels = [("0", "0", 1), ("r", "0", 1), ("1", "0", 1), ("1", "1", 0)]

    rows = []
    psi_start = psi0
    for pulse_name, h, duration in (
        ("atom_pi_1", h_atom, t_atom),
        ("ion_pi", h_ion, t_ion),
        ("atom_pi_2", h_atom, t_atom),
    ):
        for tau in np.linspace(0.0, 1.0, samples_per_pulse):
            psi = expm(-1j * h * (tau * duration)) @ psi_start
            for atom, ion, phonon in labels:
                amp = psi[basis_index(ATOM[atom], ION[ion], phonon, params)]
                rows.append(
                    {
                        "initial": state_label(*initial),
                        "pulse": pulse_name,
                        "tau": tau,
                        "state": state_label(atom, ion, phonon),
                        "real": amp.real,
                        "imag": amp.imag,
                        "probability": abs(amp) ** 2,
                    }
                )
        psi_start = expm(-1j * h * duration) @ psi_start
    return pd.DataFrame(rows)


def scan_fidelity(params: PhysicalParams, omega_atom_ghz: np.ndarray) -> pd.DataFrame:
    rows = []
    for omega in omega_atom_ghz:
        p = PhysicalParams(**{**asdict(params), "omega_atom_ghz": float(omega)})
        scales = interaction_scales(p)
        metrics = average_gate_fidelity(p)
        rows.append(
            {
                "omega_atom_ghz": float(omega),
                "fidelity": metrics["fidelity"],
                "leakage": metrics["leakage"],
                "U1_over_Omega_a": scales["U1_rad_s"] / p.omega_atom,
                "Delta_over_etaOmega_i": scales["Delta_rad_s"] / (p.eta_lamb_dicke * p.omega_ion),
            }
        )
    return pd.DataFrame(rows)


def plot_shifted_trap(params: PhysicalParams, out_path: Path) -> pd.DataFrame:
    distances_um = np.linspace(2.0, 5.0, 260)
    omega_bar, x_shift = shifted_trap(params, distances_um * 1.0e-6)
    df = pd.DataFrame(
        {
            "distance_um": distances_um,
            "omega_bar_mhz": omega_bar / (2.0 * np.pi * 1.0e6),
            "x_shift_um": x_shift * 1.0e6,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.8), constrained_layout=True)
    axes[0].plot(df["distance_um"], df["omega_bar_mhz"], lw=2.2)
    axes[0].axhline(params.omega_i_mhz, color="0.3", ls="--", lw=1.1)
    axes[0].axvline(params.ion_atom_distance_um, color="tab:red", ls=":", lw=1.1)
    axes[0].set_xlabel("ion-atom distance x0 (um)")
    axes[0].set_ylabel("shifted phonon frequency (MHz)")
    axes[1].semilogy(df["distance_um"], np.abs(df["x_shift_um"]), lw=2.2)
    axes[1].axvline(params.ion_atom_distance_um, color="tab:red", ls=":", lw=1.1)
    axes[1].set_xlabel("ion-atom distance x0 (um)")
    axes[1].set_ylabel("|equilibrium shift| (um)")
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return df


def plot_amplitude_traces(trace_df: pd.DataFrame, out_path: Path) -> None:
    initial_order = list(dict.fromkeys(trace_df["initial"]))
    pulse_order = ["atom_pi_1", "ion_pi", "atom_pi_2"]
    fig, axes = plt.subplots(len(initial_order), 3, figsize=(12.4, 5.7), sharex=True, sharey=True, constrained_layout=True)
    for row, initial in enumerate(initial_order):
        for col, pulse in enumerate(pulse_order):
            ax = axes[row, col]
            data = trace_df[(trace_df["initial"] == initial) & (trace_df["pulse"] == pulse)]
            for state, group in data.groupby("state"):
                if group["probability"].max() < 1.0e-3:
                    continue
                ax.plot(group["tau"], group["real"], lw=1.6, label=f"Re {state}")
                ax.plot(group["tau"], group["imag"], lw=1.2, ls="--", label=f"Im {state}")
            ax.axhline(0.0, color="0.85", lw=0.8)
            ax.set_title(f"{initial}, {pulse}")
            ax.set_xlabel("dimensionless time")
            if col == 0:
                ax.set_ylabel("amplitude")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, fontsize=8)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_fidelity_scan(scan_df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.8, 4.0), constrained_layout=True)
    ax.plot(scan_df["omega_atom_ghz"], scan_df["fidelity"], marker="o", ms=3.5, lw=2.0)
    ax.axhline(0.9, color="0.35", ls="--", lw=1.0)
    ax.set_xlabel("atomic Rabi frequency Omega_a / 2pi (GHz)")
    ax.set_ylabel("average CNOT fidelity")
    ax.set_ylim(0.82, 1.005)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def run(params: PhysicalParams, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    scales = interaction_scales(params)

    shift_df = plot_shifted_trap(params, output_dir / "fig2_shifted_trap.png")
    shift_df.to_csv(output_dir / "fig2_shifted_trap.csv", index=False)

    traces = pd.concat(
        [
            segment_trace(params, ("0", "0", params.n_phonon_initial)),
            segment_trace(params, ("1", "0", params.n_phonon_initial)),
        ],
        ignore_index=True,
    )
    traces.to_csv(output_dir / "fig3_amplitude_traces.csv", index=False)
    plot_amplitude_traces(traces, output_dir / "fig3_amplitude_traces.png")

    scan = scan_fidelity(params, np.linspace(0.2, 5.0, 49))
    scan.to_csv(output_dir / "fig4_fidelity_scan.csv", index=False)
    plot_fidelity_scan(scan, output_dir / "fig4_fidelity_scan.png")

    u_eff = computational_matrix(params)
    pd.DataFrame(np.real(u_eff)).to_csv(output_dir / "projected_gate_real.csv", index=False, header=False)
    pd.DataFrame(np.imag(u_eff)).to_csv(output_dir / "projected_gate_imag.csv", index=False, header=False)

    summary = {
        "paper": "arXiv:2602.19222v2, Ion-atom two-qubit quantum gate based on phonon blockade",
        "params": asdict(params),
        "scales": {
            **scales,
            "omega_bar_mhz": scales["omega_bar_rad_s"] / (2.0 * np.pi * 1.0e6),
            "Delta_mhz": scales["Delta_rad_s"] / (2.0 * np.pi * 1.0e6),
            "U1_ghz": scales["U1_rad_s"] / (2.0 * np.pi * 1.0e9),
            "U2_mhz": scales["U2_rad_s"] / (2.0 * np.pi * 1.0e6),
            "etaOmega_i_mhz": params.eta_lamb_dicke * params.omega_ion / (2.0 * np.pi * 1.0e6),
        },
        "model_notes": (
            "The parameter rydberg_phonon_leakage_scale is an effective strength for the nonideal "
            "U1-driven phonon leakage during Rydberg pi pulses in Eq. (10)-(11). The default 0.46 "
            "calibrates the compact truncated model to the paper's roughly 90% fidelity at "
            "Omega_a/2pi = 1 GHz. Set it to 0 for the ideal fast-pulse limit."
        ),
        "metrics_at_default": average_gate_fidelity(params),
        "fidelity_scan_max": scan.loc[scan["fidelity"].idxmax()].to_dict(),
        "outputs": [
            "fig2_shifted_trap.png",
            "fig3_amplitude_traces.png",
            "fig4_fidelity_scan.png",
        ],
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "outputs")
    parser.add_argument("--distance-um", type=float, default=PhysicalParams.ion_atom_distance_um)
    parser.add_argument("--omega-atom-ghz", type=float, default=PhysicalParams.omega_atom_ghz)
    parser.add_argument("--omega-ion-mhz", type=float, default=PhysicalParams.omega_ion_mhz)
    parser.add_argument("--eta", type=float, default=PhysicalParams.eta_lamb_dicke)
    parser.add_argument("--n-phonon-max", type=int, default=PhysicalParams.n_phonon_max)
    parser.add_argument("--rydberg-phonon-leakage-scale", type=float, default=PhysicalParams.rydberg_phonon_leakage_scale)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    params = PhysicalParams(
        ion_atom_distance_um=args.distance_um,
        omega_atom_ghz=args.omega_atom_ghz,
        omega_ion_mhz=args.omega_ion_mhz,
        eta_lamb_dicke=args.eta,
        n_phonon_max=args.n_phonon_max,
        rydberg_phonon_leakage_scale=args.rydberg_phonon_leakage_scale,
    )
    summary = run(params, args.output_dir)
    print(json.dumps(summary["scales"], indent=2))
    print(json.dumps(summary["metrics_at_default"], indent=2))
    print(f"Saved outputs to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
