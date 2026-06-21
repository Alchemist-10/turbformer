import torch
import torch.nn as nn
from pytorch_msssim import ms_ssim


class OAMCompositeLoss(nn.Module):
    def __init__(self, lambda_l1=1.0, lambda_ssim=0.2, lambda_oam=0.1):
        super().__init__()
        self.l1 = lambda_l1
        self.ssim = lambda_ssim
        self.oam = lambda_oam
        self.l1_loss = nn.L1Loss()

    def diff_oam_loss(self, pred_cplx, gt_cplx, num_points=128, radius=0.5):
        """
        Extracts an azimuthal ring using grid_sample to compute OAM spectrum via FFT.
        pred_cplx: Tensor (B, 2, H, W) representing Real and Imag.
        """
        B, C, H, W = pred_cplx.shape
        device = pred_cplx.device

        # Create polar coordinates in range [-1, 1] for grid_sample
        theta = torch.linspace(0, 2 * torch.pi, num_points, device=device)
        x = radius * torch.cos(theta)
        y = radius * torch.sin(theta)

        # Grid shape: (B, 1, num_points, 2)
        grid = torch.stack([x, y], dim=-1).view(1, 1, num_points, 2).expand(B, -1, -1, -1)

        # Sample ring: shape (B, 2, 1, num_points)
        pred_ring = torch.nn.functional.grid_sample(pred_cplx, grid, align_corners=True).squeeze(2)
        gt_ring = torch.nn.functional.grid_sample(gt_cplx, grid, align_corners=True).squeeze(2)

        # Convert to complex numbers
        pred_complex_ring = torch.complex(pred_ring[:, 0], pred_ring[:, 1])
        gt_complex_ring = torch.complex(gt_ring[:, 0], gt_ring[:, 1])

        # 1D FFT over the azimuthal axis to get the OAM spectrum
        pred_fft = torch.fft.fft(pred_complex_ring, dim=1)
        gt_fft = torch.fft.fft(gt_complex_ring, dim=1)

        # Compare Mode power
        return nn.MSELoss()(torch.abs(pred_fft), torch.abs(gt_fft))

    def forward(self, pred, target):
        loss_l1 = self.l1_loss(pred, target)

        # Reconstruct intensities: I = Re^2 + Im^2
        pred_int = pred[:, 0:1] ** 2 + pred[:, 1:2] ** 2
        target_int = target[:, 0:1] ** 2 + target[:, 1:2] ** 2

        # SSIM Loss
        loss_ssim = 1 - ms_ssim(pred_int, target_int, data_range=1.0, size_average=True)

        # Differentiable OAM Spectrum Loss
        loss_oam = self.diff_oam_loss(pred, target)

        return (self.l1 * loss_l1) + (self.ssim * loss_ssim) + (self.oam * loss_oam)