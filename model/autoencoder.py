import torch
from torch import nn


class Encoder(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(Encoder, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True,
                            dropout=dropout, bidirectional=False)

    def forward(self, X):
        outputs, (hidden, cell) = self.lstm(X)
        return hidden, cell


class Decoder(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(Decoder, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True,
                            dropout=dropout, bidirectional=False)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(hidden_size, input_size)

    def forward(self, X, hidden, cell):
        output, (hidden, cell) = self.lstm(X, (hidden, cell))
        output = self.relu(output)
        output = self.fc(output)
        return output, hidden, cell


class AutoEncoder(nn.Module):
    def __init__(self, input_size=8, hidden_size=4, num_layers=2, dropout=0.1):
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder(input_size, hidden_size, num_layers, dropout)
        self.decoder = Decoder(input_size, hidden_size, num_layers, dropout)

    def forward(self, X):
        batch_size, sequence_length, feature_length = X.size()
        hidden, cell = self.encoder(X)

        output = []
        temp_input = torch.zeros((batch_size, 1, feature_length), dtype=torch.float).to(X.device)
        for t in range(sequence_length):
            temp_input, hidden, cell = self.decoder(temp_input, hidden, cell)
            output.append(temp_input)

        inv_idx = torch.arange(sequence_length - 1, -1, -1).long()  # 翻转
        output = torch.cat(output, dim=1)[:, inv_idx, :]
        return output
