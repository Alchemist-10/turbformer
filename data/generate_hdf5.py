import argparse
import math
import os
from pathlib import Path

import h5py
import numpy as np
from scipy.special import genlaguerre


def cart2pol(x, y):
    return np.hypot(x, y), np.arctan2(y, x)


def lg_beam(X, Y, l, p, w0=0.03):
    """
    Paraxial Laguerre-Gaussian beam (unnormalized but stable for simulation).

    """
    r, phi = cart2pol(X, Y)
    L_pl = genlaguerre(p, abs(l))

    # Stable normalization-like factor
    norm = math.sqrt(2 * math.factorial(p) / (math.pi * math.factorial(p + abs(l))))
    rho = np.sqrt(2.0) * r / w0

    amplitude = (
        (norm / w0) * (rho ** abs(l)) * L_pl(2 * r**2 / w0**2) * np.exp(-(r**2) / w0**2)
    )
    phase = np.exp(1j * l * phi)
    return amplitude * phase


def kolmogorov_phase_screen(r0, N, dx, rng):
    """
    Fourier-domain Kolmogorov phase screen.
    r0: Fried parameter (smaller => stronger turbulence)
    """
    if r0 <= 0:
        raise ValueError(
            "r0 must be positive. Use np.inf or skip turbulence for a clean sample."
        )

    fx = np.fft.fftfreq(N, d=dx)
    fy = np.fft.fftfreq(N, d=dx)
    FX, FY = np.meshgrid(fx, fy, indexing="xy")
    f = np.sqrt(FX**2 + FY**2)

    # Avoid singularity at DC
    f[0, 0] = np.inf

    # Kolmogorov PSD (simulation-oriented)
    psd = 0.023 * (r0 ** (-5.0 / 3.0)) * (f ** (-11.0 / 3.0))
    psd[~np.isfinite(psd)] = 0.0
    psd[0, 0] = 0.0

    noise = rng.normal(size=(N, N)) + 1j * rng.normal(size=(N, N))
    phase_freq = noise * np.sqrt(psd)

    phase = np.fft.ifft2(phase_freq).real
    phase -= phase.mean()

    # The raw inverse FFT amplitude depends on the discrete grid scaling.
    # Normalize only the random shape, then restore Fried-parameter strength.
    aperture_m = N * dx
    phase_std = (aperture_m / r0) ** (5.0 / 6.0)
    phase = phase / (phase.std() + 1e-8) * phase_std
    return phase


def fresnel_propagate(field, wavelength, z, dx):
    """
    Fresnel propagation in Fourier domain.
    This is the missing step that makes the intensity actually change.
    """
    N = field.shape[0]
    fx = np.fft.fftfreq(N, d=dx)
    fy = np.fft.fftfreq(N, d=dx)
    FX, FY = np.meshgrid(fx, fy, indexing="xy")

    H = np.exp(-1j * np.pi * wavelength * z * (FX**2 + FY**2))
    propagated = np.fft.ifft2(np.fft.fft2(field) * H)
    return propagated


def normalize_complex(field):
    scale = np.max(np.abs(field)) + 1e-8
    return field / scale


def generate_dataset(
    out_path,
    num_samples,
    r0_range,
    image_size=224,
    aperture_m=0.10,
    wavelength=632.8e-9,
    z_range=(50.0, 50.0),
    seed=42,
):
    rng = np.random.default_rng(seed)

    N = image_size
    L = aperture_m
    dx = L / N

    x = np.linspace(-L / 2, L / 2, N)
    y = np.linspace(-L / 2, L / 2, N)
    X, Y = np.meshgrid(x, y, indexing="xy")

    l_choices = np.array([-8, -7, -6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6, 7, 8])
    p_choices = np.array([0, 1, 2])

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()

    print(f"[Dataset] Creating {out_path} ...", flush=True)

    with h5py.File(out_path, "w") as f:
        ds_in = f.create_dataset(
            "input",
            shape=(num_samples, 3, N, N),
            dtype=np.float32,
            compression="gzip",
            compression_opts=4,
            shuffle=True,
            chunks=(1, 3, N, N),
        )
        ds_gt = f.create_dataset(
            "ground_truth",
            shape=(num_samples, 2, N, N),
            dtype=np.float32,
            compression="gzip",
            compression_opts=4,
            shuffle=True,
            chunks=(1, 2, N, N),
        )
        # metadata = [l, p, r0, z]
        ds_meta = f.create_dataset(
            "metadata",
            shape=(num_samples, 4),
            dtype=np.float32,
            compression="gzip",
            compression_opts=4,
            shuffle=True,
            chunks=(1024, 4),
        )

        for i in range(num_samples):
            l_mode = int(rng.choice(l_choices))
            p_mode = int(rng.choice(p_choices))
            r0 = float(rng.uniform(*r0_range))
            z = float(rng.uniform(*z_range))

            clean_field = lg_beam(X, Y, l=l_mode, p=p_mode)
            clean_field = normalize_complex(clean_field)

            phase_screen = kolmogorov_phase_screen(r0=r0, N=N, dx=dx, rng=rng)

            # Apply turbulence then propagate
            field_after_turbulence = clean_field * np.exp(1j * phase_screen)
            distorted_field = fresnel_propagate(
                field_after_turbulence,
                wavelength=wavelength,
                z=z,
                dx=dx,
            )
            distorted_field = normalize_complex(distorted_field)

            # Input: intensity + complex channels of the distorted field
            ds_in[i, 0] = np.abs(distorted_field) ** 2
            ds_in[i, 1] = np.real(distorted_field)
            ds_in[i, 2] = np.imag(distorted_field)

            # Target: clean complex field
            ds_gt[i, 0] = np.real(clean_field)
            ds_gt[i, 1] = np.imag(clean_field)

            ds_meta[i] = np.array([l_mode, p_mode, r0, z], dtype=np.float32)

            if i % 500 == 0 or i == num_samples - 1:
                print(f" -> {out_path.name}: {i + 1}/{num_samples}", flush=True)

    print(f"[Success] Saved {out_path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_samples", type=int, default=12000)
    parser.add_argument("--val_samples", type=int, default=1500)
    parser.add_argument("--test_samples", type=int, default=1500)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--aperture_m", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    generate_dataset(
        out_path="train.h5",
        num_samples=args.train_samples,
        r0_range=(0.15, 0.3),
        image_size=args.image_size,
        aperture_m=args.aperture_m,
        seed=args.seed,
    )
    generate_dataset(
        out_path="val.h5",
        num_samples=args.val_samples,
        r0_range=(0.15, 0.3),
        image_size=args.image_size,
        aperture_m=args.aperture_m,
        seed=args.seed + 1,
    )
    generate_dataset(
        out_path="test_ood.h5",
        num_samples=args.test_samples,
        r0_range=(0.15, 0.35),
        image_size=args.image_size,
        aperture_m=args.aperture_m,
        seed=args.seed + 2,
    )
