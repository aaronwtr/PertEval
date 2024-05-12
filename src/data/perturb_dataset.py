import torch

import numpy as np

from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

from src.utils.spectra import get_splits


class PerturbData(Dataset):
    def __init__(self, adata, data_path, spectral_parameter, spectra_params, stage):
        self.data_name = data_path.split('/')[-1]
        self.data_path = data_path
        self.spectra_params = spectra_params
        self.stage = stage

        if self.data_name == "norman":
            highly_variable_genes = adata.var_names[adata.var['highly_variable']]
            hv_pert_adata = adata[:, highly_variable_genes]

            single_gene_mask = [True if "+" not in name else False for name in hv_pert_adata.obs['perturbation_name']]
            self.sghv_pert_adata = hv_pert_adata[single_gene_mask, :]
            self.sghv_pert_adata.obs['condition'] = self.sghv_pert_adata.obs['perturbation_name'].replace('control', 'ctrl')

            ctrl_adata = self.sghv_pert_adata[self.sghv_pert_adata.obs['condition'] == 'ctrl', :]
            pert_adata = self.sghv_pert_adata[self.sghv_pert_adata.obs['condition'] != 'ctrl', :]

            train, test, pert_list = get_splits.spectra(spectral_parameter)

            pert_list_idx = [i for i in range(len(pert_list))]
            pert_list_dict = {pert_list[i]: i for i in range(len(pert_list))}
            train_perts = [pert_list[i] for i in train]
            test_perts = [pert_list[i] for i in test]

            train_target = pert_adata[pert_adata.obs['condition'].isin(train_perts), :]
            test_target = pert_adata[pert_adata.obs['condition'].isin(test_perts), :]

            num_perts = len(pert_list)

            pert_one_hot_ref = torch.eye(num_perts)[pert_list_idx]

            all_perts_train = train_target.obs['condition'].values
            all_perts_train_idx = [pert_list_dict[pert] for pert in all_perts_train]

            all_perts_test = test_target.obs['condition'].values
            all_perts_test_idx = [pert_list_dict[pert] for pert in all_perts_test]

            one_hot_perts_train = pert_one_hot_ref[torch.tensor(all_perts_train_idx)]
            one_hot_perts_test = pert_one_hot_ref[torch.tensor(all_perts_test_idx)]

            num_train_cells = one_hot_perts_train.shape[0]
            num_test_cells = one_hot_perts_test.shape[0]
            num_ctrl_cells = ctrl_adata.shape[0]

            train_input_expr = ctrl_adata[np.random.randint(0, num_ctrl_cells, num_train_cells), :].X.toarray()
            test_input_expr = ctrl_adata[np.random.randint(0, num_ctrl_cells, num_test_cells), :].X.toarray()

            raw_X_train = np.concatenate((train_input_expr, one_hot_perts_train), axis=1)
            raw_train_target = train_target.X.toarray()

            X_train, X_val, train_targets, val_targets = train_test_split(raw_X_train,
                                                                          raw_train_target,
                                                                          test_size=0.2)
            self.X_train = torch.from_numpy(X_train)
            self.train_target = torch.from_numpy(train_targets)
            self.X_val = torch.from_numpy(X_val)
            self.val_target = torch.from_numpy(val_targets)
            self.X_test = torch.from_numpy(np.concatenate((test_input_expr, one_hot_perts_test), axis=1))
            self.test_target = torch.from_numpy(test_target.X.toarray())

    def __getitem__(self, index):
        if self.stage == "train":
            return self.X_train[index], self.train_target[index]
        elif self.stage == "val":
            return self.X_val[index], self.val_target[index]
        else:
            return self.X_test[index], self.test_target[index]

    def __len__(self):
        if self.stage == "train":
            return len(self.X_train)
        elif self.stage == "val":
            return len(self.X_val)
        else:
            return len(self.X_test)
