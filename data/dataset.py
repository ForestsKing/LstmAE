from torch.utils import data


class Dataset(data.Dataset):
    def __init__(self, data, nstep):
        self.data = data
        self.nstep = nstep

    def __getitem__(self, index):
        X = self.data[index:index + self.nstep, :]
        y = self.data[index:index + self.nstep, :]
        return X, y

    def __len__(self):
        return len(self.data) - self.nstep + 1
