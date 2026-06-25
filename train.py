import torch
import argparse
from torch.utils.data import DataLoader
from data.dataset import PhaseAwareDataset
from models.swin_restorer import OAMRestoreNet
from engine.losses import OAMCompositeLoss
from torch.optim.lr_scheduler import CosineAnnealingLR


def train_one_epoch(epoch, epochs, model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    num_batches = len(dataloader)
    
    print(f"[Train] Starting Epoch {epoch + 1}/{epochs}... Processing batches.", flush=True)

    for batch_idx, (inputs, gts, _) in enumerate(dataloader):
        inputs, gts = inputs.to(device), gts.to(device)
        optimizer.zero_grad()

        preds = model(inputs)
        loss = criterion(preds, gts)

        loss.backward()
        optimizer.step()
        
        current_loss = loss.item()
        total_loss += current_loss

        # Live feedback update every 50 batches
        if batch_idx % 50 == 0:
            running_avg = total_loss / (batch_idx + 1)
            print(f"[Train] Epoch {epoch + 1}/{epochs} | Batch {batch_idx}/{num_batches} | Current Batch Loss: {current_loss:.4f} | Running Avg: {running_avg:.4f}", flush=True)

    return total_loss / num_batches


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--no_oam', action='store_true', help='Ablation: Disable OAM Loss')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume training from')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Init] Using execution device: {device}", flush=True)

    # Force num_workers=0 to prevent multi-processing deadlock freezes inside server containers
    train_ds = PhaseAwareDataset('train.h5')
    val_ds = PhaseAwareDataset('val.h5')
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    print("[Init] Loading Swin-Restorer Backbone and weight checkpoints...", flush=True)
    model = OAMRestoreNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    criterion = OAMCompositeLoss(lambda_oam=0.0 if args.no_oam else 0.1).to(device)

    best_loss = float('inf')
    start_epoch = 0
    total_epochs = args.epochs

    # Resume logic: restore all state needed to continue training exactly after
    # the last completed checkpoint epoch, while treating --epochs as additional
    # epochs to run after the checkpoint.
    if args.resume:
        print(f"[Checkpoint] Resuming from {args.resume}", flush=True)
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        best_loss = checkpoint["best_loss"]
        last_completed_epoch = checkpoint["epoch"]
        start_epoch = last_completed_epoch
        total_epochs = last_completed_epoch + args.epochs
        print(f"[Checkpoint] Loaded checkpoint from epoch {last_completed_epoch}", flush=True)
        print(f"[Checkpoint] Continuing training from epoch {start_epoch + 1}", flush=True)
    else:
        print("[Checkpoint] Starting training from scratch.", flush=True)
    
    print("[Init] Starting pipeline execution loop.", flush=True)
    for epoch in range(start_epoch, total_epochs):
        train_loss = train_one_epoch(epoch, total_epochs, model, train_loader, optimizer, criterion, device)
        scheduler.step()

        # Simple Validation Check
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, gts, _ in val_loader:
                inputs, gts = inputs.to(device), gts.to(device)
                preds = model(inputs)
                val_loss += criterion(preds, gts).item()
        val_loss /= len(val_loader)

        # Clear summary statement for Gradio at the end of every epoch
        print(f"[Train] === Epoch {epoch + 1} Complete === Final Train Loss: {train_loss:.4f} | Validation Loss: {val_loss:.4f}", flush=True)

        if val_loss < best_loss:
            best_loss = val_loss
            print(f"[Checkpoint] Validation Loss improved. Saving weights to 'best_model.pth'", flush=True)
            torch.save(model.state_dict(), 'best_model.pth')

        # Resume logic: save the full training state so optimizer, scheduler,
        # best validation loss, and epoch progress can be restored later.
        torch.save({
            "epoch": epoch + 1,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_loss": best_loss,
        }, "checkpoint.pth")


if __name__ == '__main__':
    main()
