import numpy as np
import h5py
import os
from scipy.special import genlaguerre
import argparse



def cart2pol(x, y):
    return np.sqrt(x ** 2 + y ** 2), np.arctan2(y, x)

def lg_beam(X, Y, l, p, w0=0.03):
    r, phi = cart2pol(X, Y)
    L_pl = genlaguerre(p, abs(l))
    term1 = np.sqrt(2 * np.math.factorial(p) / (np.pi * np.math.factorial(p + abs(l))))
    term2 = (1 / w0) * ((np.sqrt(2) * r / w0) ** abs(l))
    term3 = L_pl(2 * r ** 2 / w0 ** 2) * np.exp(-r ** 2 / w0 ** 2)
    phase = np.exp(1j * l * phi)
    return term1 * term2 * term3 * phase


def kolmogorov_phase_screen(r0, N, L):
    delta = L / N
    fx = np.fft.fftfreq(N, d=delta)
    fy = np.fft.fftfreq(N, d=delta)
    FX, FY = np.meshgrid(fx, fy)
    f = np.sqrt(FX ** 2 + FY ** 2)
    f[0, 0] = 1  # Avoid division by zero

    # Kolmogorov power spectrum
    psd = 0.023 * r0 ** (-5 / 3) * f ** (-11 / 3)
    psd[0, 0] = 0

    noise = np.random.normal(0, 1, (N, N)) + 1j * np.random.normal(0, 1, (N, N))
    phase_freq = noise * np.sqrt(psd) / (delta * N)
    phase_screen = np.real(np.fft.ifft2(phase_freq))
    return phase_screen


def generate_dataset(num_samples, split_name, r0_range):
    N, L = 224, 0.1  # 224x224 to natively match Swin tiny patches
    x = np.linspace(-L / 2, L / 2, N)
    y = np.linspace(-L / 2, L / 2, N)
    X, Y = np.meshgrid(x, y)

    filename = f"{split_name}.h5"
    with h5py.File(filename, 'w') as f:
        ds_in = f.create_dataset("input", (num_samples, 3, N, N), dtype=np.float32)
        ds_gt = f.create_dataset("ground_truth", (num_samples, 2, N, N), dtype=np.float32)
        ds_meta = f.create_dataset("metadata", (num_samples, 2), dtype=np.float32)  # [mode, r0]

        for i in range(num_samples):
            l_mode = np.random.choice([-3, -2, -1, 1, 2, 3])
            r0 = np.random.uniform(*r0_range)

            clean_field = lg_beam(X, Y, l=l_mode, p=0)
            phase_screen = kolmogorov_phase_screen(r0, N, L)
            distorted_field = clean_field * np.exp(1j * phase_screen)

            # Normalize to avoid vanishing gradients
            norm_factor = np.max(np.abs(clean_field))
            clean_field /= norm_factor
            distorted_field /= norm_factor

            # Pack inputs (I, Re, Im)
            ds_in[i, 0] = np.abs(distorted_field) ** 2
            ds_in[i, 1] = np.real(distorted_field)
            ds_in[i, 2] = np.imag(distorted_field)

            # Pack GT (Re, Im)
            ds_gt[i, 0] = np.real(clean_field)
            ds_gt[i, 1] = np.imag(clean_field)
            ds_meta[i] = [l_mode, r0]

            if i % 500 == 0:
                print(f"{split_name}: {i}/{num_samples} generated.")


if __name__ == "__main__":
    generate_dataset(16000, "train", (0.02, 0.1))
    generate_dataset(2000, "val", (0.02, 0.1))
    generate_dataset(2000, "test_ood", (0.005, 0.015))  # Strong unseen turbulence