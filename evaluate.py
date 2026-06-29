import torch
import time
import numpy as np
import os
import argparse
from data.dataset import PhaseAwareDataset
from models.swin_restorer import OAMRestoreNet
from torch.utils.data import DataLoader
from scipy.ndimage import map_coordinates


KAGGLE_DATASET = "/kaggle/input/datasets/akshay10alchemist/vortex-beam-224"
OOD_PATH = os.path.join(KAGGLE_DATASET, "test_ood.h5")
if not os.path.isfile(OOD_PATH):
    OOD_PATH = "test_ood.h5"

MODEL_PATH = "best_model_epoch34.pth"


def compute_oam_purity(field, target_l, radius=0.5, num_points=128):
    """
    Compute OAM mode purity from an azimuthal decomposition on a sampled ring.
    radius is normalized to the image half-width, matching grid_sample coordinates.
    """
    H, W = field.shape
    theta = np.linspace(0.0, 2.0 * np.pi, num_points)

    x_norm = radius * np.cos(theta)
    y_norm = radius * np.sin(theta)
    x = (x_norm + 1.0) * (W - 1) / 2.0
    y = (y_norm + 1.0) * (H - 1) / 2.0

    real_ring = map_coordinates(np.real(field), [y, x], order=1, mode="constant", cval=0.0)
    imag_ring = map_coordinates(np.imag(field), [y, x], order=1, mode="constant", cval=0.0)
    ring = real_ring + 1j * imag_ring

    spectrum = np.abs(np.fft.fft(ring)) ** 2
    spectrum_sum = np.sum(spectrum) + 1e-12
    spectrum = spectrum / spectrum_sum

    mode_idx = int(target_l) % num_points
    return spectrum[mode_idx]


def compute_metrics(model, dataloader, device, radius=0.5, num_points=128):
    model.eval()
    latencies = []
    purity_scores = []

    with torch.no_grad():
        for inputs, gts, meta in dataloader:
            inputs = inputs.to(device)
            target_modes = meta[:, 0].numpy()

            start = time.time()
            preds = model(inputs)
            if torch.cuda.is_available(): torch.cuda.synchronize()
            latencies.append(time.time() - start)

            # Reconstruct complex field for OAM purity
            pred_np = preds.cpu().numpy()
            for b in range(preds.shape[0]):
                field = pred_np[b, 0] + 1j * pred_np[b, 1]
                purity_scores.append(
                    compute_oam_purity(
                        field,
                        target_l=target_modes[b],
                        radius=radius,
                        num_points=num_points,
                    )
                )

    avg_latency = (np.mean(latencies) / inputs.shape[0]) * 1000  # ms per sample
    print(f"Inference Latency: {avg_latency:.2f} ms")
    print(f"Average Mode Purity: {np.mean(purity_scores) * 100:.2f}%")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--radius", type=float, default=0.5)
    parser.add_argument("--num_points", type=int, default=128)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = OAMRestoreNet().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    ood_loader = DataLoader(PhaseAwareDataset(OOD_PATH), batch_size=8)
    compute_metrics(model, ood_loader, device, radius=args.radius, num_points=args.num_points)
