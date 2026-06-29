import torch
import numpy as np
import matplotlib.pyplot as plt
import h5py

from models.swin_restorer import OAMRestoreNet

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "best_model_epoch34.pth"
DATA_PATH = "test_ood.h5"

# --------------------------
# Load model
# --------------------------
model = OAMRestoreNet().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# --------------------------
# Pick one sample
# --------------------------
with h5py.File(DATA_PATH, "r") as f:
    idx = 0  # change this to any sample

    inp = f["input"][idx]
    gt = f["ground_truth"][idx]
    meta = f["metadata"][idx]

# --------------------------
# Run inference
# --------------------------
x = torch.tensor(inp).unsqueeze(0).float().to(DEVICE)

with torch.no_grad():
    pred = model(x)

pred = pred.squeeze(0).cpu().numpy()

# --------------------------
# Convert to intensity images
# --------------------------
input_intensity = inp[0]

pred_complex = pred[0] + 1j * pred[1]
pred_intensity = np.abs(pred_complex) ** 2

gt_complex = gt[0] + 1j * gt[1]
gt_intensity = np.abs(gt_complex) ** 2

# normalize for visualization
pred_intensity /= pred_intensity.max() + 1e-8
gt_intensity /= gt_intensity.max() + 1e-8

# --------------------------
# Plot
# --------------------------
fig, ax = plt.subplots(1, 3, figsize=(15, 5))

ax[0].imshow(input_intensity, cmap="inferno")
ax[0].set_title("Turbulent Input")
ax[0].axis("off")

ax[1].imshow(pred_intensity, cmap="inferno")
ax[1].set_title("Restored Output")
ax[1].axis("off")

ax[2].imshow(gt_intensity, cmap="inferno")
ax[2].set_title("Ground Truth")
ax[2].axis("off")

plt.tight_layout()
plt.show()

print("Metadata:")
print(f"l = {int(meta[0])}")
print(f"p = {int(meta[1])}")
print(f"r0 = {meta[2]:.4f}")
print(f"z = {meta[3]:.2f}")