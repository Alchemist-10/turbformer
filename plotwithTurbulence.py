import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.special import genlaguerre


def cart2pol(x, y):
    return np.sqrt(x ** 2 + y ** 2), np.arctan2(y, x)


def lg_beam(X, Y, l, p, w0=0.03):
    r, phi = cart2pol(X, Y)
    L_pl = genlaguerre(p, abs(l))
    term1 = np.sqrt(2 * math.factorial(p) / (np.pi * math.factorial(p + abs(l))))
    term2 = (1 / w0) * ((np.sqrt(2) * r / w0) ** abs(l))
    term3 = L_pl(2 * r ** 2 / w0 ** 2) * np.exp(-r ** 2 / w0 ** 2)
    phase = np.exp(1j * l * phi)
    return term1 * term2 * term3 * phase


def kolmogorov_phase_screen(r0, N, dx, rng):
    if np.isinf(r0):
        return np.zeros((N, N), dtype=np.float64)
    if r0 <= 0:
        raise ValueError("r0 must be positive. Use np.inf for the no-turbulence reference.")

    fx = np.fft.fftfreq(N, d=dx)
    fy = np.fft.fftfreq(N, d=dx)
    FX, FY = np.meshgrid(fx, fy, indexing="xy")
    f = np.sqrt(FX ** 2 + FY ** 2)

    f[0, 0] = np.inf
    psd = 0.023 * (r0 ** (-5.0 / 3.0)) * (f ** (-11.0 / 3.0))
    psd[~np.isfinite(psd)] = 0.0
    psd[0, 0] = 0.0

    noise = rng.normal(size=(N, N)) + 1j * rng.normal(size=(N, N))
    phase_freq = noise * np.sqrt(psd)
    phase = np.fft.ifft2(phase_freq).real

    phase -= phase.mean()
    aperture_m = N * dx
    phase_std = (aperture_m / r0) ** (5.0 / 6.0)
    phase = phase / (phase.std() + 1e-8) * phase_std
    return phase


def fresnel_propagate(field, wavelength, z, dx):
    N = field.shape[0]
    fx = np.fft.fftfreq(N, d=dx)
    fy = np.fft.fftfreq(N, d=dx)
    FX, FY = np.meshgrid(fx, fy, indexing="xy")

    H = np.exp(-1j * np.pi * wavelength * z * (FX ** 2 + FY ** 2))
    return np.fft.ifft2(np.fft.fft2(field) * H)


def normalize_complex(field):
    scale = np.max(np.abs(field)) + 1e-8
    return field / scale


def simulate_sample(X, Y, l_mode, p_mode, r0, z, N, dx, wavelength, rng):
    clean_field = lg_beam(X, Y, l=l_mode, p=p_mode)
    clean_field = normalize_complex(clean_field)

    phase_screen = kolmogorov_phase_screen(r0=r0, N=N, dx=dx, rng=rng)
    field_after_turbulence = clean_field * np.exp(1j * phase_screen)

    distorted_field = fresnel_propagate(
        field_after_turbulence,
        wavelength=wavelength,
        z=z,
        dx=dx,
    )
    distorted_field = normalize_complex(distorted_field)

    clean_intensity = np.abs(clean_field) ** 2
    distorted_intensity = np.abs(distorted_field) ** 2

    return clean_field, distorted_field, clean_intensity, distorted_intensity, phase_screen


def plot_random_turbulence_demo():
    rng = np.random.default_rng(42)

    # Match generator settings
    N = 128
    L = 0.10
    dx = L / N
    wavelength = 632.8e-9

    x = np.linspace(-L / 2, L / 2, N)
    y = np.linspace(-L / 2, L / 2, N)
    X, Y = np.meshgrid(x, y, indexing="xy")

    # Fixed beam and propagation, vary only r0 to see effect clearly
    l_mode = -3
    p_mode = 1
    z = 50.0

    r0_values = [np.inf,0.4,0.3,0.2, 0.150]

    fig, axes = plt.subplots(len(r0_values), 4, figsize=(18, 4 * len(r0_values)))

    if len(r0_values) == 1:
        axes = np.expand_dims(axes, axis=0)

    for i, r0 in enumerate(r0_values):
        clean_field, distorted_field, clean_intensity, distorted_intensity, phase_screen = simulate_sample(
            X=X,
            Y=Y,
            l_mode=l_mode,
            p_mode=p_mode,
            r0=r0,
            z=z,
            N=N,
            dx=dx,
            wavelength=wavelength,
            rng=rng,
        )

        clean_real = np.real(clean_field)
        distorted_real = np.real(distorted_field)

        # Column 1: clean intensity
        im0 = axes[i, 0].imshow(clean_intensity, cmap="inferno")
        axes[i, 0].set_title(f"Clean Intensity\n(l={l_mode}, p={p_mode})")
        axes[i, 0].axis("off")
        fig.colorbar(im0, ax=axes[i, 0], fraction=0.046, pad=0.04)

        # Column 2: clean real
        im1 = axes[i, 1].imshow(clean_real, cmap="bwr")
        axes[i, 1].set_title("Clean Real")
        axes[i, 1].axis("off")
        fig.colorbar(im1, ax=axes[i, 1], fraction=0.046, pad=0.04)

        # Column 3: distorted intensity
        im2 = axes[i, 2].imshow(distorted_intensity, cmap="inferno")
        r0_label = "no turbulence" if np.isinf(r0) else f"r0={r0:.3f} m"
        axes[i, 2].set_title(f"Distorted Intensity\n({r0_label}, z={z:.1f} m)")
        axes[i, 2].axis("off")
        fig.colorbar(im2, ax=axes[i, 2], fraction=0.046, pad=0.04)

        im3 = axes[i, 3].imshow(phase_screen, cmap="twilight")
        axes[i, 3].set_title(f"Phase Screen\nstd={phase_screen.std():.3f} rad")
        axes[i, 3].axis("off")
        fig.colorbar(im3, ax=axes[i, 3], fraction=0.046, pad=0.04)

    plt.suptitle(
        "TurbFormer Dataset Sample Visualization\n(Fixed beam, varying r0)",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )
    plt.tight_layout()
    plt.savefig("turbulence_comparison.png", dpi=300, bbox_inches="tight")
    plt.show()

    print("[Success] Saved turbulence_comparison.png")


if __name__ == "__main__":
    plot_random_turbulence_demo()
