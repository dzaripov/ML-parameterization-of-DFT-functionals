import h5py    
import numpy as np    
import torch
from torch import nn
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
import random
from tqdm.auto import tqdm
# from tqdm.notebook import tqdm
from SVWN3 import f_svwn3
from sklearn.preprocessing import StandardScaler
from NN_models import NN_2_256, NN_8_256, NN_8_64

with h5py.File('C6H6_mgae109.h5', "r") as f:
  y = np.array(f["ener"][:])
  X_raw = np.array(f["grid"][:])

X = X_raw[:, 4:-1]
train_size = int(X.shape[0] * 0.8)
X_train = X[:train_size]
X_test = X[train_size:]

y_train_dist = [0.0310907, 0.01554535, 
                3.72744,   7.06042,
                12.9352,   18.0578,
                -0.10498,  -0.32500,
                0.0310907,  0.01554535,  -1/(6*np.pi**2),
                13.0720,    20.1231,      1.06835,
                42.7198,   101.578,      11.4813,
                -0.409286,  -0.743294,   -0.228344,
                1]

nconstants = len(y_train_dist)

y_train = np.log1p(np.tile(y_train_dist, [X_train.shape[0],1]))
y_test = np.log1p(np.tile(y_train_dist, [X_test.shape[0],1]))


scaler = StandardScaler()
X_train = np.log1p(X_train)
X_train = scaler.fit_transform(X_train)

X_test = np.log1p(X_test)
X_test = scaler.transform(X_test)

def set_random_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
set_random_seed(42)

device = torch.device('cuda:0') if torch.cuda.is_available else torch.device('cpu')
device


class Dataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X).float()
        self.y = torch.tensor(y).float()
        
    def __getitem__(self, i):
        return self.X[i], self.y[i]
    
    def __len__(self):
        return len(self.X)

BATCH_SIZE = 1024

train_set = Dataset(X=X_train, y=y_train)
train_dataloader = torch.utils.data.DataLoader(train_set, 
                                            batch_size=BATCH_SIZE, shuffle=True) #, num_workers=4) # , num_workers=1

test_set = Dataset(X=X_test, y=y_test)
test_dataloader = torch.utils.data.DataLoader(test_set, 
                                            batch_size=BATCH_SIZE) #, num_workers=4) #, num_workers=1


model = NN_8_64().to(device)
# model.load_state_dict(torch.load('predoptimized_3.param'))

optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, betas=(0.9, 0.999))
criterion = nn.MSELoss()
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)


def train(model, criterion, optimizer, scheduler, train_loader, test_loader, n_epochs=200):
    train_loss_mse = []
    train_loss_mae = []
    test_loss_mse = []
    test_loss_mae = []


    for epoch in range(n_epochs):
        print(epoch+1)
        # train
        model.train()
        progress_bar = tqdm(train_loader)


        train_mse_losses_per_epoch = []
        train_mae_losses_per_epoch = []
        for X_batch, y_batch in progress_bar:


            X_batch, y_batch = X_batch.to(device), y_batch.to(device) # переехали на гпу
            predictions = model(X_batch) # смотрим че есть
            loss = criterion(predictions, y_batch) # оцениваем масштабы бедствия
            loss.backward() # обновляем градиенты
            optimizer.step() # делаем шаг градиентного спуска 
            optimizer.zero_grad()
            # print(predictions, y_batch)
            train_mse_losses_per_epoch.append(loss.item())
            train_mae_losses_per_epoch.append(mean_absolute_error(predictions.cpu().detach(), y_batch.cpu().detach()))
        train_loss_mse.append(np.mean(train_mse_losses_per_epoch))
        train_loss_mae.append(np.mean(train_mae_losses_per_epoch))


        #test
        model.eval()
        test_mse_losses_per_epoch = []
        test_mae_losses_per_epoch = []
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                preds = model(X_batch)
                loss = criterion(preds, y_batch)
                test_mse_losses_per_epoch.append(loss.item())
                test_mae_losses_per_epoch.append(mean_absolute_error(preds.cpu().detach(), y_batch.cpu().detach()))
        test_loss_mse.append(np.mean(test_mse_losses_per_epoch))
        test_loss_mae.append(np.mean(test_mae_losses_per_epoch))
        print(f'train RMSE Loss = {train_loss_mse[epoch] ** 0.5:.8f}')
        print(f'train MAE Loss = {train_loss_mae[epoch]:.8f}')
        print(f'test RMSE Loss = {test_loss_mse[epoch] ** 0.5:.8f}')
        print(f'test MAE Loss = {test_loss_mae[epoch]:.8f}')

    scheduler.step()
    return train_loss_mse, train_loss_mae, test_loss_mse, test_loss_mae, preds[0].cpu().detach().numpy()
        # print(np.array(train_accumulated_loss_mae).sum())
        # print('train RMSE Loss = {:.4f}'.format((train_accumulated_loss_mse. / len(X_train)) ** 0.5))
        # print('train MAE Loss = {:.4f}'.format((train_accumulated_loss_mae.sum() / len(X_train))))
        # print('test RMSE Loss = {:.4f}'.format((test_accumulated_loss_mse.sum() / len(X_test))) ** 0.5)


train_loss_mse, train_loss_mae, test_loss_mse, test_loss_mae, preds = train(model, criterion, optimizer, 
                                                                            scheduler, train_dataloader,
                                                                            test_dataloader, n_epochs=200)

# print(train_loss_mse, train_loss_mae, test_loss_mse, test_loss_mae)
print('predicted coef', '\n',np.expm1(preds))
print('exact coef', '\n', np.array(y_train_dist))

torch.save(model.state_dict(), 'predoptimized_3.param')