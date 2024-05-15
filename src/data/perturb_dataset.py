import torch
import os
import anndata

import numpy as np
import scanpy as sc

from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from src.utils.spectra import get_splits


class PerturbData(Dataset):
    def __init__(self, adata, data_path, spectral_parameter, spectra_params, stage):
        self.data_name = data_path.split('/')[-1]
        self.data_path = data_path
        self.spectra_params = spectra_params
        self.stage = stage

        if self.data_name == "norman":
            single_gene_mask = [True if "+" not in name else False for name in adata.obs['perturbation_name']]
            sg_pert_adata = adata[single_gene_mask, :]
            sg_pert_adata.obs['condition'] = sg_pert_adata.obs['perturbation_name'].replace('control', 'ctrl')

            genes = sg_pert_adata.var.index.to_list()
            genes_and_ctrl = genes + ['ctrl']

            # we remove the cells with perts that are not in the genes because we need gene expression values
            # to generate an in-silico perturbation embedding
            sg_pert_adata = sg_pert_adata[sg_pert_adata.obs['condition'].isin(genes_and_ctrl), :]

            train, test, pert_list = get_splits.spectra(sg_pert_adata,
                                                        self.data_path,
                                                        self.spectra_params,
                                                        spectral_parameter
                                                        )

            print(f"Norman dataset has {len(pert_list)} perturbations in common with the genes in the dataset.")

            pert_adata = sg_pert_adata[sg_pert_adata.obs['condition'] != 'ctrl', :]

            if not os.path.exists(f"{self.data_path}/full_{self.data_name}_filtered.h5ad"):
                sg_pert_adata.write(f"{self.data_path}/full_{self.data_name}_filtered.h5ad")

            if not os.path.exists(f"{self.data_path}/basal_ctrl_{self.data_name}_filtered.h5ad"):
                ctrl_adata = sg_pert_adata[sg_pert_adata.obs['condition'] == 'ctrl', :]
                ctrl_X = ctrl_adata.X.toarray()
                basal_ctrl_X = np.zeros((pert_adata.shape[0], ctrl_X.shape[1]))
                subset_size = 500

                for cell in tqdm(range(pert_adata.shape[0])):
                    subset = ctrl_X[np.random.choice(ctrl_X.shape[0], subset_size), :]
                    basal_ctrl_X[cell, :] = subset.mean(axis=0)

                basal_ctrl_adata = anndata.AnnData(X=basal_ctrl_X, obs=pert_adata.obs, var=ctrl_adata.var)

                # noinspection PyTypeChecker
                basal_ctrl_adata.write(f"{self.data_path}/basal_ctrl_{self.data_name}_filtered.h5ad")
            else:
                basal_ctrl_adata = sc.read_h5ad(f"{self.data_path}/basal_ctrl_{self.data_name}_filtered.h5ad")

            pert_list_idx = [i for i in range(len(pert_list))]
            pert_list_dict = {pert_list[i]: i for i in range(len(pert_list))}
            train_perts = [pert_list[i] for i in train]
            test_perts = [pert_list[i] for i in test]

            # todo: predict just on HVGs. That is output will be of size (num_cells, num_HVGs)
            highly_variable_genes = adata.var_names[adata.var['highly_variable']]
            hv_pert_adata = adata[:, highly_variable_genes]

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
            num_ctrl_cells = basal_ctrl_adata.shape[0]

            train_input_expr = basal_ctrl_adata[np.random.randint(0, num_ctrl_cells, num_train_cells), :].X.toarray()
            test_input_expr = basal_ctrl_adata[np.random.randint(0, num_ctrl_cells, num_test_cells), :].X.toarray()

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

        if self.data_name == "repogle_rpe1":
            # TODO: filter on HVGs after saving!!
            if not os.path.exists(f"{self.data_path}/{self.data_name}_filtered.h5ad"):
                adata.layers["counts"] = adata.X.copy()
                sc.pp.normalize_total(adata)
                sc.pp.log1p(adata)
                adata.write(f"{self.data_path}/{self.data_name}_filtered.h5ad", compression='gzip')
            else:
                adata = sc.read_h5ad(f"{self.data_path}/{self.data_name}_filtered.h5ad")

            sghv_pert_adata = adata[:, adata.var['highly_variable']]
            sghv_pert_adata.obs['condition'] = sghv_pert_adata.obs['perturbation'].replace('control', 'ctrl')

            ctrl_adata = sghv_pert_adata[sghv_pert_adata.obs['condition'] == 'ctrl', :]
            pert_adata = sghv_pert_adata[sghv_pert_adata.obs['condition'] != 'ctrl', :]

            # sample 1000 cells from the sghv_pert_adata
            sghv_pert_adata = sghv_pert_adata[np.random.choice(sghv_pert_adata.shape[0], 100, replace=False), :]

            train, test, pert_list = get_splits.spectra(sghv_pert_adata,
                                                        self.data_path,
                                                        self.spectra_params,
                                                        spectral_parameter
                                                        )

            # sc.pp.highly_variable_genes(adata, inplace=True, n_top_genes=5000)

            print('joe')

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
