import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from data.dataset import Dataset


def test(X_test, model, scaler, threshold, batch_size=16, nstep=64):
    X_test = scaler.transform(X_test.values)
    dataset_test = Dataset(X_test, nstep=nstep)
    dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    predict, src = [], []
    model.eval()
    for i, (batch_x, batch_y) in enumerate(dataloader_test):
        batch_x = batch_x.float().to(device)
        output = model(batch_x)
        predict.extend(output.detach().cpu().numpy())
        src.extend(batch_x.detach().cpu().numpy())

    residuals = pd.Series(np.sum(np.mean(np.abs(src - np.array(predict)), axis=1), axis=1))
    head = np.zeros(nstep-1-int((nstep-1)/2))
    tail = np.zeros(int((nstep-1)/2))
    prediction = pd.Series((residuals > threshold).astype(int).values).fillna(0).values
    prediction = np.hstack([head, prediction, tail])
    return prediction
