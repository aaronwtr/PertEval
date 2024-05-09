import torch

from torch.utils.data import Dataset


class PerturbData(Dataset):
    def __init__(self, data, labels, gene_names, test_source=False):
        self.data = data
        self.labels = labels
        self.gene_names = gene_names

        self.test_source = test_source

        x = self.data
        y = self.labels

        self.features = torch.tensor(x, dtype=torch.float32)
        self.labels = torch.tensor(y, dtype=torch.float32)

    def __getitem__(self, index):
        if self.test_source is False:
            return self.features[index], self.labels[index]
        else:
            return self.features[index], self.labels[index], self.test_source

    def __len__(self):
        return len(self.labels)