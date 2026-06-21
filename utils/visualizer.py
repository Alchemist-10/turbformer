import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from models.swin_restorer import OAMRestoreNet
from data.dataset import PhaseAwareDataset
from torch.utils.data import DataLoader


def plot_results(model_path, data_path, output_dir="plots"):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load Model and Data
    model = OAMRestoreNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    dataset = PhaseAwareDataset(data_path)
    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    inputs, gts, meta = next(iter(loader))

    with torch.no_grad():
        preds = model(inputs.to(device)).cpu()

    # Convert Complex fields to Intensity (I = Re^2 + Im^2)
    inputs_np = inputs.numpy()
    distorted_intensity = inputs_np[:, 0, :, :]  # Channel 0 is already intensity
    gt_intensity = gts[:, 0] ** 2 + gts[:, 1] ** 2
    pred_intensity = preds[:, 0] ** 2 + preds[:, 1] ** 2

    fig, axes = plt.subplots(4, 3, figsize=(10, 12))
    fig.suptitle("Atmospheric Turbulence Compensation Results", fontsize=16)

    for i in range(4):
        # Distorted Input
        axes[i, 0].imshow(distorted_intensity[i], cmap='inferno')
        axes[i, 0].set_title(f"Distorted Input\nMode: {int(meta[i, 0])}, r0: {meta[i, 1]:.3f}")
        axes[i, 0].axis('off')

        # Restored Output
        axes[i, 1].imshow(pred_intensity[i], cmap='inferno')
        axes[i, 1].set_title("Model Restoration")
        axes[i, 1].axis('off')

        # Ground Truth
        axes[i, 2].imshow(gt_intensity[i], cmap='inferno')
        axes[i, 2].set_title("Clean Target")
        axes[i, 2].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "restoration_grid.png"), dpi=300)
    print(f"Saved visualization to {output_dir}/restoration_grid.png")


if __name__ == "__main__":
    plot_results("best_model.pth", "test_ood.h5")