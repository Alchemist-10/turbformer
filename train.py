import torch
import argparse
from torch.utils.data import DataLoader
from data.dataset import PhaseAwareDataset
from models.swin_restorer import OAMRestoreNet
from engine.losses import OAMCompositeLoss
from torch.optim.lr_scheduler import CosineAnnealingLR


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for inputs, gts, _ in dataloader:
        inputs, gts = inputs.to(device), gts.to(device)
        optimizer.zero_grad()

        preds = model(inputs)
        loss = criterion(preds, gts)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--no_oam', action='store_true', help='Ablation: Disable OAM Loss')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_ds = PhaseAwareDataset('train.h5')
    val_ds = PhaseAwareDataset('val.h5')
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = OAMRestoreNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    criterion = OAMCompositeLoss(lambda_oam=0.0 if args.no_oam else 0.1).to(device)

    best_loss = float('inf')
    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
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

        print(f"Epoch {epoch + 1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')


if __name__ == '__main__':
    main()