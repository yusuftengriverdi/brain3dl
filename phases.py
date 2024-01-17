import torch
import torch.nn as nn
import torch.optim as optim

try: 
    from losses import *
except:
    from .losses import *
from tqdm import tqdm
import datetime
from metrics import haussdorf, rel_abs_vol_dif

def one_hot_mask(labels,  num_classes = 3):

    b, c, h, w, d = labels.shape

    y = torch.zeros(b, num_classes, h, w, d, dtype=torch.int64)
    
    for idx in range(1, 4):
        y[:, idx-1, :, :, :] = (labels == idx).squeeze(1).to(torch.int64)

    return y.float()

def one_hot_mask_2d(labels,  num_classes = 3):

    b, c, h, w = labels.shape

    y = torch.zeros(b, num_classes, h, w, dtype=torch.int64)
    
    for idx in range(1, 4):
        y[:, idx-1, :, :] = (labels == idx).squeeze(1).to(torch.int64)

    return y.float()

def train_one_epoch(model, train_loader, criterion, optimizer, device, scheduler):
    model.train()

    running_loss = 0.0
    running_dice = 0.0
    running_ravd = 0.0
    running_hd = 0.0

    pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc="Initial Description")

    for batch, item in pbar:
        # for 3d
        # X = item['volume'].unsqueeze(1)
        # y = item['label'].unsqueeze(1)
        X = item['volume']
        y = item['label']
    
        del item

        # import matplotlib.pyplot as plt
        # plt.subplot(241)
        # plt.imshow(X[1, 0])
        # plt.subplot(242)
        # plt.imshow(X[2, 0])
        # plt.subplot(243)
        # plt.imshow(X[3, 0])
        # plt.subplot(244)
        # plt.imshow(X[4, 0])
        # plt.subplot(245)
        # plt.imshow(y[1, 0])
        # plt.subplot(246)
        # plt.imshow(y[2, 0])
        # plt.subplot(247)
        # plt.imshow(y[3, 0])
        # plt.subplot(248)
        # plt.imshow(y[4, 0])
        # plt.show()

        X, y = X.to(device), y.to(device)

        optimizer.zero_grad()

        yhat = model(X)
        del X 

        if len(y.shape) == 5: 
            y = one_hot_mask(y)
            running_dice += compute_dice_score(yhat, y)
        else:
            y = one_hot_mask_2d(y)
            running_dice += compute_dice_score_2d(yhat, y)

        loss = criterion(yhat, y)
        loss.backward()
        optimizer.step()
        # running_hd += haussdorf(yhat, y)

        del yhat, y
        running_loss += loss.item()
        # running_hd += compute_hd(yhat, y)
        pbar.set_description(f"Training loss: {running_loss / (batch + 1)}, Dice: {running_dice/ (batch+1)}, Hd: {running_hd / (batch + 1)} ")
    
    if not scheduler is None:
        scheduler.step()
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
            
            X = item['volume']
            y = item['label']
            del item 

            X, y = X.to(device), y.to(device)

            yhat = model(X)
            del X 

            if len(y.shape)== 5:
                y = one_hot_mask(y)
                running_dice += compute_dice_score(yhat, y)
            else:
                y = one_hot_mask_2d(y)
                running_dice += compute_dice_score_2d(yhat, y)

            loss = criterion(yhat, y)
            # running_hd += haussdorf(y, yhat)

            del yhat, y 

            running_loss += loss.item()

            # running_ravd += compute_ravd(yhat, y)
            # running_hd += compute_hd(yhat, y)
            pbar.set_description(f"Val loss: {running_loss / (batch + 1)}, Dice: {running_dice / (batch + 1)}, Hd: {running_hd / (batch + 1)}")

    running_loss /= len(val_loader)
    running_dice /= len(val_loader)
    running_ravd /= len(val_loader)
    running_hd /= len(val_loader)

    return running_loss, running_dice, running_ravd, running_hd


def train_and_validate(model, train_loader, val_loader, num_epochs, learning_rate=0.01, device='cuda'):
    model.to(device)

    date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    criterion = SoftDiceLoss()
    #criterion = nn.MSELoss(reduction='mean') # Dice increasing 0.40 ep2 0.01 adam , loss is kinda stable??? 
    # criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

    best_val_dice = -0.0

    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")

        # Training
        train_loss, train_dice, train_ravd, train_hd = train_one_epoch(model, train_loader, criterion, optimizer, device, scheduler=None)
        print(f"Training Loss: {train_loss:.4f}")

        # Validation
        val_loss, val_dice, val_ravd, val_hd = validate(model, val_loader, criterion, device)
        print(f"Validation Loss: {val_loss:.4f}")

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save(model, f'runs/best_model_val_dice_{date}.pth')


    print("\nTraining complete!")

# Example usage:
# Assuming you have train_loader, val_loader, and model ready
# train_and_validate(model, train_loader, val_loader, num_epochs=5)
