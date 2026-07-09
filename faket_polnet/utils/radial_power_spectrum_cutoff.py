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


def compare_power_spectra(style_path, faket_path, mode, save_path=None, freq_cutoff=None):
    """
    Compare a style and a faket (style-transferred) tomogram via their radial power
    spectra. The style is center-cropped to match the faket tomogram's shape.

    Parameters:
        style_path (str): Path to the style (reference) tomogram.
        faket_path (str): Path to the faket (style-transferred) tomogram.
        mode (str): FTT computation mode: "2d" or "3d".
        freq_cutoff (float): If set, restrict the log-L2 to normalized frequencies
            at or below this value, renormalized within the retained band.

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
    freq = np.linspace(0, 1, len(rps1))

    keep = freq <= freq_cutoff if freq_cutoff is not None else np.ones(len(freq), dtype=bool)
    p1 = rps1[keep] / rps1[keep].sum()
    p2 = rps2[keep] / rps2[keep].sum()
    eps = 1e-12
    log_l2 = float(np.sqrt(np.mean((np.log10(p1 + eps) - np.log10(p2 + eps)) ** 2)))

    if save_path is not None:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.semilogy(freq, rps1, label=Path(style_path).stem, color="steelblue")
        ax.semilogy(freq, rps2, label=Path(faket_path).stem.replace("_faket", ""), color="coral")
        if freq_cutoff is not None:
            ax.axvline(freq_cutoff, color="gray", linestyle="--", linewidth=1)
        ax.set_xlabel("Spatial Frequency")
        ax.set_ylabel("Log(Power)")
        ax.set_title(f"Power Spectrum {mode.upper()} (Log-L2 = {log_l2:.4f})")
        ax.legend()
        plt.tight_layout()
        fig.savefig(save_path, dpi=150)
        plt.close(fig)

    return log_l2
