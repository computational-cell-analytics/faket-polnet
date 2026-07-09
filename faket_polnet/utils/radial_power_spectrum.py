import numpy as np
import matplotlib.pyplot as plt
import mrcfile
from pathlib import Path

from .utils import center_crop


def compute_radial_power_spectrum(arr, mode):
    """
    Radially-averaged power spectrum: per-slice mean (2D) or full volume (3D).

    Parameters:
        arr (np.ndarray): 2D image or 3D volume.
        mode (str): FTT computation mode: "2d" or "3d".

    Returns:
        np.ndarray: 1D radial power profile from zero frequency to the Nyquist
            of the shortest axis.
    """
    if mode == "2d":
        return np.mean([compute_radial_power_spectrum(s, mode="3d") for s in arr], axis=0)

    ps = np.abs(np.fft.fftshift(np.fft.fftn(arr))) ** 2
    center = np.array(ps.shape) // 2
    coords = np.indices(ps.shape)
    r = np.sqrt(sum((c - ctr) ** 2 for c, ctr in zip(coords, center))).astype(int)
    r_max = int(center.min())
    power = np.bincount(r.ravel(), weights=ps.ravel())
    counts = np.bincount(r.ravel())
    return (power / counts)[:r_max]


def compare_power_spectra(style_path, faket_path, mode, save_path=None):
    """
    Compare a style and a faket (style-transferred) tomogram via their radial power
    spectra. The style is center-cropped to match the faket tomogram's shape.

    Parameters:
        style_path (str): Path to the style (reference) tomogram.
        faket_path (str): Path to the faket (style-transferred) tomogram.
        mode (str): FTT computation mode: "2d" or "3d".

    Returns:
        float: log-L2 distance between the two normalized power spectra.
    """
    with mrcfile.open(style_path, permissive=True) as mrc:
        style = np.copy(mrc.data).astype(np.float32)
    with mrcfile.open(faket_path, permissive=True) as mrc:
        faket = np.copy(mrc.data).astype(np.float32)
    style = center_crop(style, faket.shape)

    rps1 = compute_radial_power_spectrum(style, mode)
    rps2 = compute_radial_power_spectrum(faket, mode)
    rps1 = rps1 / rps1.sum()
    rps2 = rps2 / rps2.sum()

    eps = 1e-12
    log_l2 = float(np.sqrt(np.mean((np.log10(rps1 + eps) - np.log10(rps2 + eps)) ** 2)))

    if save_path is not None:
        freq = np.linspace(0, 1, len(rps1))
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.semilogy(freq, rps1, label=Path(style_path).stem, color="steelblue")
        ax.semilogy(freq, rps2, label=Path(faket_path).stem.replace("_faket", ""), color="coral")
        ax.set_xlabel("Spatial Frequency")
        ax.set_ylabel("Log(Power)")
        ax.set_title(f"Power Spectrum {mode.upper()} (Log-L2 = {log_l2:.4f})")
        ax.legend()
        plt.tight_layout()
        fig.savefig(save_path, dpi=150)
        plt.close(fig)

    return log_l2


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


def compare_intensity_distributions(style_path, faket_path, bins=200, save_path=None):
    """
    Compare the intensity distributions of a style and a faket (style-transferred)
    tomogram. The style is center-cropped to match the faket tomogram's shape.

    Parameters:
        style_path (str): Path to the style (reference) tomogram.
        faket_path (str): Path to the faket (style-transferred) tomogram.
        bins (int): Number of histogram bins.

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

    fig, ax = plt.subplots(figsize=(7, 5))
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
