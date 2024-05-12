from src.utils.spectra.spectra import Spectra
from src.utils.spectra.dataset import SpectraDataset

import numpy as np
from tqdm import tqdm
from gears.pertdata import PertData


class PerturbGraphData(SpectraDataset):
    def parse(self, pert_data):
        if isinstance(pert_data, PertData):
            self.adata = pert_data.adata
        else:
            self.adata = pert_data
        self.control_expression = self.adata[self.adata.obs['condition'] == 'ctrl'].X.toarray().mean(axis=0)
        return [p for p in self.adata.obs['condition'].unique() if p != 'ctrl']

    def get_mean_logfold_change(self, perturbation):
        perturbation_expression = self.adata[self.adata.obs['condition'] == perturbation].X.toarray().mean(axis=0)
        logfold_change = np.nan_to_num(np.log2(perturbation_expression + 1) - np.log2(self.control_expression + 1))
        return logfold_change

    def sample_to_index(self, sample):
        if not hasattr(self, 'index_to_sequence'):
            print("Generating index to sequence")
            self.index_to_sequence = {}
            for i in tqdm(range(len(self))):
                x = self.__getitem__(i)
                self.index_to_sequence['-'.join(list(x))] = i

        return self.index_to_sequence[sample]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        perturbation = self.samples[idx]
        return self.get_mean_logfold_change(perturbation)


class SPECTRAPerturb(Spectra):
    def spectra_properties(self, sample_one, sample_two):
        return -np.linalg.norm(sample_one - sample_two)

    def cross_split_overlap(self, train, test):
        average_similarity = []

        for i in test:
            for j in train:
                average_similarity.append(self.spectra_properties(i, j))

        return np.mean(average_similarity)
