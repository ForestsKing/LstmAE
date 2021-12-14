import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from data.dataset import Dataset
from utils.earlystoping import EarlyStopping


def train(X_train, scaler, model, lr=0.005, epochs=100, batch_size=16, patience=10, Q=0.99, validation_split=0.2,
          nstep=64, path='log/', showloss=True):
    X_test = X_train
    X_train = X_test[:-int(len(X_test) * validation_split)]
    X_valid = X_test[-int(len(X_test) * validation_split):]

    X_train = scaler.fit_transform(X_train.values)
    X_valid = scaler.transform(X_valid.values)
    X_test = scaler.transform(X_test.values)

    dataset_train = Dataset(X_train, nstep=nstep)
    dataset_valid = Dataset(X_valid, nstep=nstep)
    dataset_test = Dataset(X_test, nstep=nstep)

    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    dataloader_valid = DataLoader(dataset_valid, batch_size=batch_size, shuffle=True)
    dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 0.95 ** ((epoch - 1) // 10))
    early_stopping = EarlyStopping(patience=patience, verbose=False, path=path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    avg_train_losses, avg_valid_losses = [], []
    for epoch in range(epochs):
        train_losses, valid_losses = [], []

        model.train()
        for i, (batch_x, batch_y) in enumerate(dataloader_train):
            optimizer.zero_grad()
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)

            output = model(batch_x)
            loss = criterion(output, batch_y)

            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        for i, (batch_x, batch_y) in enumerate(dataloader_valid):
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)

            output = model(batch_x)
            loss = criterion(output, batch_y)
            valid_losses.append(loss.item())

        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)

        if showloss:
            epoch_len = len(str(epochs))
            print_msg = (f'[{epoch + 1:>{epoch_len}}/{epochs:>{epoch_len}}] ' +
                         f'train_loss: {train_loss:.5f} ' +
                         f'valid_loss: {valid_loss:.5f}')

            print(print_msg)
        early_stopping(valid_loss, model)

        if early_stopping.early_stop:
            break
        scheduler.step()

    model.load_state_dict(torch.load(path + 'model_checkpoint.pkl'))

    predict, src = [], []
    model.eval()
    for i, (batch_x, batch_y) in enumerate(dataloader_test):
        batch_x = batch_x.float().to(device)
        output = model(batch_x)
        predict.extend(output.detach().cpu().numpy())
        src.extend(batch_x.detach().cpu().numpy())

    residuals = pd.Series(np.sum(np.mean(np.abs(src - np.array(predict)), axis=1), axis=1))
    threshold = 3 / 2 * residuals.quantile(Q)

    # joblib.dump(scaler, path + 'scaler.pkl')
    # joblib.dump(threshold, path + 'threshold.pkl')
    return model, scaler, threshold, avg_train_losses, avg_valid_losses
