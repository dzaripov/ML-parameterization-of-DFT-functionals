import torch
import gc
from tqdm.autonotebook import tqdm
from sklearn.metrics import mean_absolute_error
import numpy as np


def predopt(model, criterion, optimizer, train_loader, device, n_epochs=2, accum_iter=4):
    
    train_loss_mse = []
    train_loss_mae = []
    test_loss_mse = []
    test_loss_mae = []


    for epoch in range(n_epochs):
        print('Epoch', epoch+1)
        # train
        model.train()


        train_mse_losses_per_epoch = []
        train_mae_losses_per_epoch = []
        
        progress_bar = tqdm(train_loader)

        
        for batch_idx, (X_batch, y_batch) in enumerate(progress_bar):
            X_batch = X_batch['Grid'].to(device, non_blocking=True)
            y_batch = torch.tile(torch.Tensor(y_batch), [X_batch.shape[0],1]).to(device, non_blocking=True)
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            loss.backward()
            
            if ((batch_idx + 1) % accum_iter == 0) or (batch_idx + 1 == len(train_loader)):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                MAE = mean_absolute_error(predictions.cpu().detach(), y_batch.cpu().detach())
                MSE = loss.item()
                train_mse_losses_per_epoch.append(MSE)
                train_mae_losses_per_epoch.append(MAE)
                progress_bar.set_postfix(MAE = MAE, MSE = MSE)

                del X_batch, y_batch, predictions, loss, MAE, MSE
            gc.collect()
            torch.cuda.empty_cache()
            
        train_loss_mse.append(np.mean(train_mse_losses_per_epoch))
        train_loss_mae.append(np.mean(train_mae_losses_per_epoch))
        
        print(f'train MSE Loss = {train_loss_mse[epoch]:.8f}')
        print(f'train MAE Loss = {train_loss_mae[epoch]:.8f}')
        
    return train_loss_mse, train_loss_mae