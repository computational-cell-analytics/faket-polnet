import numpy as np
import matplotlib.pyplot as plt
import mrcfile
from pathlib import Path

from .utils import center_crop


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


def _compare_and_plot(rps1, rps2, name1, name2, title, figsize, save_path=None):
    eps = 1e-12
    rps1 = rps1 / rps1.sum()
    rps2 = rps2 / rps2.sum()
    log_l2 = float(np.sqrt(np.mean((np.log10(rps1 + eps) - np.log10(rps2 + eps)) ** 2)))

    freq = np.linspace(0, 1, len(rps1))
    fig, ax = plt.subplots(figsize=figsize)
    ax.semilogy(freq, rps1, label=name1, color="steelblue")
    ax.semilogy(freq, rps2, label=name2, color="coral")
    ax.set_xlabel("Spatial frequency")
    ax.set_ylabel("Power (log scale)")
    ax.set_title(f"{title} (log-L2 = {log_l2:.4f})")
    ax.legend()
    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return log_l2


def compare_power_spectra_2d(style_path, faket_path, figsize=(7, 5), save_path=None):
    """
    Compare a style and a faket (style-transferred) tomogram via their per-slice
    radial power spectra, averaged over z-slices. The style is center-cropped to
    match the faket tomogram's shape.

    Parameters:
        style_path (str): Path to the style (reference) tomogram.
        faket_path (str): Path to the faket (style-transferred) tomogram.
        figsize (tuple): Figure size for the plot.

    Returns:
        float: L2 distance between the two log power spectra.
    """
    with mrcfile.open(style_path, permissive=True) as mrc:
        style = np.copy(mrc.data).astype(np.float32)
    with mrcfile.open(faket_path, permissive=True) as mrc:
        faket = np.copy(mrc.data).astype(np.float32)
    style = center_crop(style, faket.shape)

    name1 = Path(style_path).stem
    name2 = Path(faket_path).stem

    rps1 = np.mean([compute_radial_power_spectrum(s) for s in style], axis=0)
    rps2 = np.mean([compute_radial_power_spectrum(s) for s in faket], axis=0)

    return _compare_and_plot(rps1, rps2, name1, name2, "Power Spectrum 2D", figsize, save_path=save_path)


def compare_power_spectra_3d(style_path, faket_path, figsize=(7, 5), save_path=None):
    """
    Compare a style and a faket (style-transferred) tomogram via 3D radial power spectra. 
    The style is center-cropped to match the faket tomogram's shape.

    Parameters:
        style_path (str): Path to the style (reference) tomogram.
        faket_path (str): Path to the faket (style-transferred) tomogram.
        figsize (tuple): Figure size for the plot.

    Returns:
        float: L2 distance between the two log power spectra.
    """
    with mrcfile.open(style_path, permissive=True) as mrc:
        style = np.copy(mrc.data).astype(np.float32)
    with mrcfile.open(faket_path, permissive=True) as mrc:
        faket = np.copy(mrc.data).astype(np.float32)
    style = center_crop(style, faket.shape)

    name1 = Path(style_path).stem
    name2 = Path(faket_path).stem

    rps1 = compute_radial_power_spectrum(style)
    rps2 = compute_radial_power_spectrum(faket)

    return _compare_and_plot(rps1, rps2, name1, name2, "Power Spectrum 3D", figsize, save_path=save_path)


def compute_intensity_distribution(arr, bins=200, value_range=None):
    """
    Normalized intensity histogram of an array.

    Parameters:
        arr (np.ndarray): Input array.
        bins (int): Number of histogram bins.
        value_range (tuple): (min, max) range for the histogram.

    Returns:
        tuple: (density, bin_centers) 1D arrays.
    """
    density, edges = np.histogram(arr.ravel(), bins=bins, range=value_range, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return density, centers


def compare_intensity_distributions(style_path, faket_path, bins=200, figsize=(7, 5), save_path=None):
    """
    Compare the intensity distributions of a style and a faket (style-transferred)
    tomogram. The style is center-cropped to match the faket tomogram's shape.

    Parameters:
        style_path (str): Path to the style (reference) tomogram.
        faket_path (str): Path to the faket (style-transferred) tomogram.
        bins (int): Number of histogram bins.
        figsize (tuple): Figure size for the plot.

    Returns:
        float: Wasserstein distance between the two intensity distributions.
    """
    from scipy.stats import wasserstein_distance

    with mrcfile.open(style_path, permissive=True) as mrc:
        style = np.copy(mrc.data).astype(np.float32)
    with mrcfile.open(faket_path, permissive=True) as mrc:
        faket = np.copy(mrc.data).astype(np.float32)
    style = center_crop(style, faket.shape)

    name1 = Path(style_path).stem
    name2 = Path(faket_path).stem

    lo = min(np.percentile(style, 1), np.percentile(faket, 1))
    hi = max(np.percentile(style, 99), np.percentile(faket, 99))
    d1, c1 = compute_intensity_distribution(style, bins, (lo, hi))
    d2, c2 = compute_intensity_distribution(faket, bins, (lo, hi))
    wd = float(wasserstein_distance(c1, c2, d1, d2))

    print(f"{'stat':<8}{name1:>16}{name2:>16}")
    for label, fn in [("mean", np.mean), ("std", np.std),
                      ("p5", lambda a: np.percentile(a, 5)),
                      ("p50", np.median),
                      ("p95", lambda a: np.percentile(a, 95))]:
        print(f"{label:<8}{fn(style):>16.4f}{fn(faket):>16.4f}")

    fig, ax = plt.subplots(figsize=figsize)
    ax.fill_between(c1, d1, alpha=0.5, label=name1, color="steelblue")
    ax.fill_between(c2, d2, alpha=0.5, label=name2, color="coral")
    ax.set_xlabel("Intensity")
    ax.set_ylabel("Density")
    ax.set_title(f"Intensity Distribution (Wasserstein = {wd:.4f})")
    ax.legend()
    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return wd


def main():
    import argparse
    import json
    parser = argparse.ArgumentParser(description="Compare radial power spectra of two tomograms and save the plot.")
    parser.add_argument("--faket_path", required=True, help="Style-transferred tomogram to evaluate.")
    parser.add_argument("--snr_json", required=True, help="JSON mapping tomogram key to style.")
    parser.add_argument("--style_dir", required=True, help="Directory containing style tomograms.")
    parser.add_argument("--mode", choices=["2d", "3d", "intensity"], default="2d")
    parser.add_argument("--tag", required=True, help="Identifier for this parameter setting.")
    parser.add_argument("--png_dir", required=True, help="Directory to save the comparison plot.")
    parser.add_argument("--csv_path", default=None, help="If set, append metrics to this CSV.")
    args = parser.parse_args()

    key = Path(args.faket_path).stem
    if key.endswith("_faket"):
        key = key[:-len("_faket")]
    with open(args.snr_json) as f:
        style_name = json.load(f)[key]["style"]
    style_path = str(Path(args.style_dir) / f"{style_name}.mrc")

    png_dir = Path(args.png_dir)
    png_dir.mkdir(parents=True, exist_ok=True)
    save_path = png_dir / f"{args.tag}_{args.mode}.png"

    compare = {
        "2d": compare_power_spectra_2d,
        "3d": compare_power_spectra_3d,
        "intensity": compare_intensity_distributions,
    }[args.mode]
    score = compare(style_path, args.faket_path, save_path=str(save_path))
    print(f"Saved figure to png: {save_path}")

    if args.csv_path:
        csv_path = Path(args.csv_path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        new_file = not csv_path.exists()
        with open(csv_path, "a") as f:
            if new_file:
                f.write("tag,mode,score\n")
            f.write(f"{args.tag},{args.mode},{score:.6f}\n")


if __name__ == "__main__":
    main()
