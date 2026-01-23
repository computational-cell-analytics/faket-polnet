import sys
from pathlib import Path

from pathlib import Path
import sys
from ..faket.style_transfer.cli import main as cli_main

def style_transfer_wrapper(
    content_path: Path,
    style_paths: list[Path],
    output_path: Path,
    devices: list[str] = None,
    iterations: int = None,
    save_every: int = None,
    min_scale: int = None,
    end_scale: int = None,
    random_seed: int = 0,
    style_weights: list[float] = None,
    extra_args: dict = None,
):
    """
    Wrapper around faket.style_transfer.cli.

    Parameters:
        content_path: Path to content tomogram.
        style_paths: List of style tomograms.
        output_path: Path to save output tomogram.
        devices: List of device strings, e.g., ['cuda:0']
        iterations, save_every, min_scale, end_scale, random_seed: NST args
        style_weights: List of floats matching style_paths.
        extra_args: dict of additional CLI args for NST.
    """

    # build a fake argv to pass to CLI
    argv = ['cli.py', str(content_path)] + [str(p) for p in style_paths]
    argv += ['--output', str(output_path)]

    if devices:
        argv += ['--devices'] + devices
    if iterations is not None:
        argv += ['--iterations', str(iterations)]
    if save_every is not None:
        argv += ['--save-every', str(save_every)]
    if min_scale is not None:
        argv += ['--min-scale', str(min_scale)]
    if end_scale is not None:
        argv += ['--end-scale', str(end_scale)]
    if random_seed is not None:
        argv += ['--random-seed', str(random_seed)]
    if style_weights:
        argv += ['--style-weights'] + [str(w) for w in style_weights]

    if extra_args:
        for k, v in extra_args.items():
            argv += [f'--{k}']
            if v is not None and v is not True:
                argv += [str(v)]
            elif v is True:
                pass  # flag argument

    # temporarily replace sys.argv to call faket CLI

    print(f"DEBUG: {argv}") # TODO remove debug
    old_argv = sys.argv
    sys.argv = argv
    try:
        cli_main()
    finally:
        sys.argv = old_argv
