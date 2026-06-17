import numpy as np
import matplotlib.pyplot as plt
import mrcfile
from pathlib import Path


def compute_radial_power_spectrum(arr):
    """
    Radially-averaged power spectrum of a 2D or 3D array.

    Parameters:
        arr (np.ndarray): 2D image or 3D volume.

    Returns:
        np.ndarray: 1D radial power profile from zero frequency to the Nyquist
            of the shortest axis.
    """
    ps = np.abs(np.fft.fftshift(np.fft.fftn(arr))) ** 2
    center = np.array(ps.shape) // 2
    coords = np.indices(ps.shape)
    r = np.sqrt(sum((c - ctr) ** 2 for c, ctr in zip(coords, center))).astype(int)
    r_max = int(center.min())
    power = np.bincount(r.ravel(), weights=ps.ravel())
    counts = np.bincount(r.ravel())
    return (power / counts)[:r_max]


def _compare_and_plot(rps1, rps2, name1, name2, title, figsize, n_freq, save_path=None):
    eps = 1e-12
    rps1 = rps1 / rps1.sum()
    rps2 = rps2 / rps2.sum()
    freq = np.linspace(0, 1, n_freq)
    p1 = np.interp(freq, np.linspace(0, 1, len(rps1)), rps1)
    p2 = np.interp(freq, np.linspace(0, 1, len(rps2)), rps2)
    log_l2 = float(np.sqrt(np.mean((np.log10(p1 + eps) - np.log10(p2 + eps)) ** 2)))

    fig, ax = plt.subplots(figsize=figsize)
    ax.semilogy(np.linspace(0, 1, len(rps1)), rps1, label=name1, color="steelblue")
    ax.semilogy(np.linspace(0, 1, len(rps2)), rps2, label=name2, color="coral")
    ax.set_xlabel("Spatial frequency (normalized)")
    ax.set_ylabel("Normalized power (log scale)")
    ax.set_title(f"{title} (log-L2 = {log_l2:.4f})")
    ax.legend()
    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return log_l2


def compare_power_spectra_2d(tomo1_path, tomo2_path, figsize=(7, 5), n_freq=256, save_path=None):
    """
    Compare two tomograms via their per-slice radial power spectra,
    averaged over z-slices.

    Parameters:
        tomo1_path (str): Path to the first tomogram.
        tomo2_path (str): Path to the second tomogram.
        figsize (tuple): Figure size for the plot.
        n_freq (int): Number of points on the shared normalized-frequency grid.

    Returns:
        float: L2 distance between the two log power spectra.
    """
    with mrcfile.open(tomo1_path, permissive=True) as mrc:
        tomo1 = np.copy(mrc.data).astype(np.float32)
    with mrcfile.open(tomo2_path, permissive=True) as mrc:
        tomo2 = np.copy(mrc.data).astype(np.float32)

    name1 = Path(tomo1_path).stem
    name2 = Path(tomo2_path).stem

    rps1 = np.mean([compute_radial_power_spectrum(s) for s in tomo1], axis=0)
    rps2 = np.mean([compute_radial_power_spectrum(s) for s in tomo2], axis=0)

    return _compare_and_plot(rps1, rps2, name1, name2, "Power Spectrum 2D", figsize, n_freq, save_path=save_path)


def compare_power_spectra_3d(tomo1_path, tomo2_path, figsize=(7, 5), n_freq=256, save_path=None):
    """
    Compare two tomograms via their full 3D radial power spectra.

    Parameters:
        tomo1_path (str): Path to the first tomogram.
        tomo2_path (str): Path to the second tomogram.
        figsize (tuple): Figure size for the plot.
        n_freq (int): Number of points on the shared normalized-frequency grid.

    Returns:
        float: L2 distance between the two log power spectra.
    """
    with mrcfile.open(tomo1_path, permissive=True) as mrc:
        tomo1 = np.copy(mrc.data).astype(np.float32)
    with mrcfile.open(tomo2_path, permissive=True) as mrc:
        tomo2 = np.copy(mrc.data).astype(np.float32)

    name1 = Path(tomo1_path).stem
    name2 = Path(tomo2_path).stem

    rps1 = compute_radial_power_spectrum(tomo1)
    rps2 = compute_radial_power_spectrum(tomo2)

    return _compare_and_plot(rps1, rps2, name1, name2, "Power Spectrum 3D", figsize, n_freq, save_path=save_path)


def main():
    import argparse
    import json
    parser = argparse.ArgumentParser(description="Compare radial power spectra of two tomograms and save the plot.")
    parser.add_argument("--styled_path", required=True, help="Style-transferred tomogram to evaluate.")
    parser.add_argument("--snr_json", required=True, help="JSON mapping tomogram key to style.")
    parser.add_argument("--style_dir", required=True, help="Directory containing style tomograms.")
    parser.add_argument("--mode", choices=["2d", "3d"], default="2d", help="2D mode computes power spectrum per slice.")
    parser.add_argument("--tag", required=True, help="Identifier for this parameter setting.")
    parser.add_argument("--png_dir", required=True, help="Directory to save the comparison plot.")
    parser.add_argument("--csv_path", default=None, help="If set, append metrics to this CSV.")
    parser.add_argument("--n_freq", type=int, default=256)
    args = parser.parse_args()

    key = Path(args.styled_path).stem
    if key.endswith("_faket"):
        key = key[:-len("_faket")]
    with open(args.snr_json) as f:
        style_name = json.load(f)[key]["style"]
    style_path = str(Path(args.style_dir) / f"{style_name}.mrc")

    png_dir = Path(args.png_dir)
    png_dir.mkdir(parents=True, exist_ok=True)
    save_path = png_dir / f"{args.tag}_{args.mode}.png"

    rad_psd = compare_power_spectra_2d if args.mode == "2d" else compare_power_spectra_3d
    log_l2 = rad_psd(style_path, args.styled_path, n_freq=args.n_freq, save_path=str(save_path))
    print(f"tag={args.tag} mode={args.mode} log_l2={log_l2:.6f} png={save_path}")

    if args.csv_path:
        csv_path = Path(args.csv_path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        new_file = not csv_path.exists()
        with open(csv_path, "a") as f:
            if new_file:
                f.write("tag,mode,log_l2\n")
            f.write(f"{args.tag},{args.mode},{log_l2:.6f}\n")


if __name__ == "__main__":
    main()
