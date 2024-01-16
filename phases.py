import torch
import torch.nn as nn
import torch.optim as optim

try: 
    from losses import DiceLoss, compute_dice_score
except:
    from .losses import DiceLoss, compute_dice_score
from tqdm import tqdm

def one_hot_mask(labels,  num_classes = 3):

    b, c, h, w, d = labels.shape

    y = torch.zeros(b, num_classes, h, w, d, dtype=torch.int64)
    
    for idx in range(1, 4):
        y[:, idx-1, :, :, :] = (labels == idx).squeeze(1).to(torch.int64)

    # Convert labels to one-hot encoding
    
    # y =nn.functional.one_hot(labels, num_classes)
    # Use scatter to fill in the one-hot encoding
    # y.scatter_(1, labels.unsqueeze(1).to(torch.int64), 1)

    return y 

def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()

    running_loss = 0.0
    running_dice = 0.0
    running_ravd = 0.0
    running_hd = 0.0

    pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc="Initial Description")

    for batch, item in pbar:
        X = item['volume'].unsqueeze(1)
        y = item['label'].unsqueeze(1)

        X, y = X.to(device), y.to(device)

        optimizer.zero_grad()

        yhat = model(X)

        y = one_hot_mask(y)
        loss = criterion(yhat, y.long())
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        
        # running_dice += compute_dice_score(yhat, y).mean()
        # running_ravd += compute_ravd(yhat, y)
        # running_hd += compute_hd(yhat, y)
        pbar.set_description(f"Training loss: {running_loss / (batch + 1)} ")

    running_loss /= len(train_loader)
    running_dice /= len(train_loader)
    running_ravd /= len(train_loader)
    running_hd /= len(train_loader)

    return running_loss, running_dice, running_ravd, running_hd


def validate(model, val_loader, criterion, device):
    model.eval()

    running_loss = 0.0
    running_dice = 0.0
    running_ravd = 0.0
    running_hd = 0.0

    with torch.no_grad():
        
        pbar = tqdm(enumerate(val_loader), total=len(val_loader), desc="Initial Description")

        for batch, item in pbar:
            
            X = item['volume'].unsqueeze(1)
            y = item['label'].unsqueeze(1)

            X, y = X.to(device), y.to(device)

            yhat = model(X)

            y = one_hot_mask(y)
            loss = criterion(yhat, y.long())

            running_loss += loss.item()

            # running_dice += compute_dice_score(yhat, y).mean()
            # running_ravd += compute_ravd(yhat, y)
            # running_hd += compute_hd(yhat, y)
            pbar.set_description(f"Val loss: {running_loss / (batch + 1)}")

    running_loss /= len(val_loader)
    running_dice /= len(val_loader)
    running_ravd /= len(val_loader)
    running_hd /= len(val_loader)

    return running_loss, running_dice, running_ravd, running_hd


def train_and_validate(model, train_loader, val_loader, num_epochs, learning_rate=0.01, device='cuda'):
    model.to(device)

    criterion = DiceLoss(reduction='sum')
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")

        # Training
        train_loss, train_dice, train_ravd, train_hd = train_one_epoch(model, train_loader, criterion, optimizer, device)
        print(f"Training Loss: {train_loss:.4f}")

        # Validation
        val_loss, val_dice, val_ravd, val_hd = validate(model, val_loader, criterion, device)
        print(f"Validation Loss: {val_loss:.4f}")

    print("\nTraining complete!")

# Example usage:
# Assuming you have train_loader, val_loader, and model ready
# train_and_validate(model, train_loader, val_loader, num_epochs=5)
