import os

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from model.autoencoder import AutoEncoder
from utils.setseed import set_seed
from utils.test import test
from utils.train import train

batch_size = 32
epochs = 40
lr = 0.005

if __name__ == '__main__':
    path = 'data/SKAB/'
    filenames = os.listdir(path)
    dfs = [pd.read_csv(path + filename, sep=';', index_col='datetime', parse_dates=True) for filename in filenames]

    set_seed(0)
    predicted_outlier = []
    true_outlier = []
    for df in tqdm(dfs):
        X_train = df[:400].drop(['anomaly', 'changepoint'], axis=1)
        X_test = df.drop(['anomaly', 'changepoint'], axis=1)

        scaler = StandardScaler()

        model = AutoEncoder()
        model, scaler, threshold, avg_train_losses, avg_valid_losses = train(X_train=X_train, scaler=scaler,
                                                                             model=model, showloss=False, epochs=100)
        prediction = test(X_test=X_test, model=model, scaler=scaler, threshold=threshold)
        predicted_outlier.extend(prediction)
        true_outlier.extend(df['anomaly'].values)

    true_outlier = np.array(true_outlier).reshape(-1)
    predicted_outlier = np.array(predicted_outlier).reshape(-1)

    print('precision_score: %.4f' % precision_score(true_outlier, predicted_outlier))
    print('recall_score_score: %.4f' % recall_score(true_outlier, predicted_outlier))
    print('f1_score: %.4f' % f1_score(true_outlier, predicted_outlier))
