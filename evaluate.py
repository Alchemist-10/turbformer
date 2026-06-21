import torch
import time
import numpy as np
from data.dataset import PhaseAwareDataset
from models.swin_restorer import OAMRestoreNet
from torch.utils.data import DataLoader


def compute_metrics(model, dataloader, device):
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
                # Center slice azimuthal FFT approximation
                center_idx = field.shape[0] // 2
                ring = field[center_idx - 20:center_idx + 20, :].mean(axis=0)  # simplified horizontal projection
                spectrum = np.abs(np.fft.fft(ring)) ** 2
                spectrum /= np.sum(spectrum)
                # target mode mapping
                l = int(target_modes[b])
                mode_idx = l if l >= 0 else len(spectrum) + l
                purity_scores.append(spectrum[mode_idx])

    avg_latency = (np.mean(latencies) / inputs.shape[0]) * 1000  # ms per sample
    print(f"Inference Latency: {avg_latency:.2f} ms")
    print(f"Average Mode Purity: {np.mean(purity_scores) * 100:.2f}%")


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = OAMRestoreNet().to(device)
    model.load_state_dict(torch.load('best_model.pth'))
    ood_loader = DataLoader(PhaseAwareDataset('test_ood.h5'), batch_size=8)
    compute_metrics(model, ood_loader, device)