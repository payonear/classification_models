from torch.utils.data import Dataset

class BlogDataset(Dataset):
    """Pytorch Dataset object created for comfortable interaction with torch DataLoaders.

    Args:
        X: numpy array with features
        y: label
    """
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return {'X': self.X[idx], 'y': self.y[idx]}
