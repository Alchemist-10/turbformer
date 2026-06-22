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
    
    print("[Init] Starting pipeline execution loop.", flush=True)
    for epoch in range(args.epochs):
        train_loss = train_one_epoch(epoch, args.epochs, model, train_loader, optimizer, criterion, device)
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


if __name__ == '__main__':
    main()