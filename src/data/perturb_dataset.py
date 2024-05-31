import torch
import os
import anndata
import gzip

import numpy as np
import scanpy as sc
import pickle as pkl
import pandas as pd

from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from hydra.errors import HydraException

from src.utils.spectra import get_splits


class PerturbData(Dataset):
    def __init__(self, adata, data_path, spectral_param, replicate, spectra_params, stage, **kwargs):
        self.data_name = data_path.split('/')[-1]
        self.data_path = data_path
        self.spectral_parameter = f"{spectral_param}_{replicate}"
        self.spectra_params = spectra_params
        self.stage = stage
        self.eval_type = kwargs.get("eval_type", None)

        if self.eval_type is not None and "_de" not in self.eval_type:
            raise ValueError("eval_type must be None or '{pert}_de'.")

        if not os.path.exists(f"{self.data_path}/input_features/train_data_{self.spectral_parameter}.pkl.gz"):
            pp_data = self.preprocess_and_featurise_norman(adata)
            self.X_train, self.train_target, self.X_val, self.val_target, self.X_test, self.test_target = pp_data
        else:
            with gzip.open(f"{self.data_path}/input_features/train_data_{self.spectral_parameter}.pkl.gz", "rb") as f:
                self.X_train, self.train_target = pkl.load(f)
            with gzip.open(f"{self.data_path}/input_features/val_data_{self.spectral_parameter}.pkl.gz", "rb") as f:
                self.X_val, self.val_target = pkl.load(f)
            with gzip.open(f"{self.data_path}/input_features/test_data_{self.spectral_parameter}.pkl.gz", "rb") as f:
                self.X_test, self.test_target = pkl.load(f)

        if self.data_name == "replogle_rpe1":
            ctrl_adata, pert_adata, train, test, pert_list = self.preprocess_replogle(adata)
            self.featurise_replogle(pert_adata, pert_list, ctrl_adata, train, test)

        if self.data_name == "replogle_k562":
            ctrl_adata, pert_adata, train, test, pert_list = self.preprocess_replogle(adata)
            self.featurise_replogle(pert_adata, pert_list, ctrl_adata, train, test)

    def preprocess_and_featurise_norman(self, adata):
        nonzero_genes = (adata.X.sum(axis=0) > 5).A1
        filtered_adata = adata[:, nonzero_genes]
        single_gene_mask = [True if "," not in name else False for name in adata.obs['guide_ids']]
        sg_adata = filtered_adata[single_gene_mask, :]
        sg_adata.obs['condition'] = sg_adata.obs['guide_ids'].replace('', 'ctrl')

        genes = sg_adata.var['gene_symbols'].to_list()
        genes_and_ctrl = genes + ['ctrl']

        # we remove the cells with perts that are not in the genes because we need gene expression values
        # to generate an in-silico perturbation embedding
        sg_pert_adata = sg_adata[sg_adata.obs['condition'].isin(genes_and_ctrl), :]

        train, test, pert_list = get_splits.spectra(sg_pert_adata,
                                                    self.data_path,
                                                    self.spectra_params,
                                                    self.spectral_parameter
                                                    )

        print(f"Norman dataset has {len(pert_list)} perturbations in common with the genes in the dataset.")

        ctrl_adata = sg_pert_adata[sg_pert_adata.obs['condition'] == 'ctrl', :]

        if not os.path.exists(f"{self.data_path}/ctrl_{self.data_name}_raw_counts.h5ad"):
            ctrl_adata.write(f"{self.data_path}/ctrl_{self.data_name}_raw_counts.h5ad", compression='gzip')

        pert_adata = sg_pert_adata[sg_pert_adata.obs['condition'] != 'ctrl', :]
        all_perts = list(set(pert_adata.obs['condition'].to_list()))
        unique_perts = list(set(pert_list))

        num_cells = ctrl_adata.shape[0]
        num_perts = len(all_perts)

        # generate embedding mask for the perturbable genes with nonzero expression values
        if not os.path.exists(f"{self.data_path}/norman_mask_df.pkl"):
            mask = np.zeros((num_cells, num_perts), dtype=bool)

            for i, pert in enumerate(all_perts):
                pert_idx = genes.index(pert)
                non_zero_indices = ctrl_adata[:, pert_idx].X.sum(axis=1).nonzero()[0]
                num_non_zeroes = len(non_zero_indices)

                if len(non_zero_indices) < 500:
                    sample_num = num_non_zeroes
                else:
                    sample_num = 500

                sampled_indices = np.random.choice(non_zero_indices, sample_num, replace=False)

                mask[sampled_indices, i] = True

            mask_df = pd.DataFrame(mask, columns=all_perts)
            mask_df.to_pickle(f"{self.data_path}/norman_mask_df.pkl")

        gene_to_ensg = dict(zip(sg_pert_adata.var['gene_symbols'], sg_pert_adata.var_names))

        if not os.path.exists(f"{self.data_path}/basal_ctrl_{self.data_name}_pp_filtered.h5ad"):
            pert_adata = sg_pert_adata[sg_pert_adata.obs['condition'] != 'ctrl', :]

            # save control_data_raw for inference with scFMs and pert_data for contextual alignment experiment
            if not os.path.exists(f"{self.data_path}/ctrl_{self.data_name}_raw_counts.h5ad"):
                ctrl_adata.write(f"{self.data_path}/ctrl_{self.data_name}_raw_counts.h5ad", compression='gzip')
            if not os.path.exists(f"{self.data_path}/pert_{self.data_name}_raw_counts.h5ad"):
                pert_adata.write(f"{self.data_path}/pert_{self.data_name}_raw_counts.h5ad", compression='gzip')

            if not os.path.exists(f"{self.data_path}/{self.data_name}_pp_ctrl_filtered.h5ad"):
                sc.pp.normalize_total(sg_pert_adata)
                sc.pp.log1p(sg_pert_adata)
                sc.pp.highly_variable_genes(sg_pert_adata, n_top_genes=2000)
                highly_variable_genes = sg_pert_adata.var_names[sg_pert_adata.var['highly_variable']]
                unique_perts_ensg = [gene_to_ensg[pert] for pert in unique_perts]
                missing_perts = list(set(unique_perts_ensg) - set(highly_variable_genes))
                combined_genes = list(set(highly_variable_genes) | set(missing_perts))
                sg_hvg_adata = sg_pert_adata[:, combined_genes]

                ctrl_adata = sg_hvg_adata[sg_hvg_adata.obs['condition'] == 'ctrl', :]
                pert_adata = sg_hvg_adata[sg_hvg_adata.obs['condition'] != 'ctrl', :]

                ctrl_adata.write(f"{self.data_path}/{self.data_name}_pp_ctrl_filtered.h5ad", compression='gzip')
                pert_adata.write(f"{self.data_path}/{self.data_name}_pp_pert_filtered.h5ad", compression='gzip')

            ctrl_X = ctrl_adata.X.toarray()
            basal_ctrl_X = np.zeros((pert_adata.shape[0], ctrl_X.shape[1]))
            subset_size = 500

            for cell in tqdm(range(pert_adata.shape[0])):
                subset_X = ctrl_X[np.random.choice(ctrl_X.shape[0], subset_size), :]
                basal_ctrl_X[cell, :] = subset_X.mean(axis=0)

            basal_ctrl_adata = anndata.AnnData(X=basal_ctrl_X, obs=pert_adata.obs, var=ctrl_adata.var)

            # noinspection PyTypeChecker
            basal_ctrl_adata.write(f"{self.data_path}/basal_ctrl_{self.data_name}_pp_filtered.h5ad", compression='gzip')
            with open(f"{self.data_path}/all_perts.pkl", "wb") as f:
                pkl.dump(all_perts, f)
        else:
            basal_ctrl_adata = sc.read_h5ad(f"{self.data_path}/basal_ctrl_{self.data_name}_pp_filtered.h5ad")
            pert_adata = sc.read_h5ad(f"{self.data_path}/{self.data_name}_pp_pert_filtered.h5ad")

        control_genes = basal_ctrl_adata.var.index.to_list()
        pert_genes = pert_adata.var.index.to_list()
        assert control_genes == pert_genes, ("Watch out! Genes in control and perturbation datasets are not the"
                                             " same, or are not indexed the same.")

        train_perts = [pert_list[i] for i in train]
        test_perts = [pert_list[i] for i in test]

        if not os.path.exists(f"{self.data_path}/test_perts_split_{self.spectral_parameter}.pkl"):
            with open(f"{self.data_path}/test_perts_{self.spectral_parameter}.pkl", "wb") as f:
                pkl.dump(test_perts, f)

        train_target = pert_adata[pert_adata.obs['condition'].isin(train_perts), :]
        test_target = pert_adata[pert_adata.obs['condition'].isin(test_perts), :]

        all_perts_train = train_target.obs['condition'].values
        all_perts_test = test_target.obs['condition'].values

        if not os.path.exists(f"{self.data_path}/pert_corrs.pkl"):
            all_gene_expression = basal_ctrl_adata.X

            pert_corrs = {}
            for pert in tqdm(unique_perts, total=len(unique_perts)):
                correlations = np.zeros(basal_ctrl_adata.shape[1])
                ensg_id = gene_to_ensg[pert]
                pert_idx = basal_ctrl_adata.var_names.get_loc(ensg_id)
                basal_expr_pert = basal_ctrl_adata.X[:, pert_idx].flatten()
                for i in range(all_gene_expression.shape[1]):
                    corr = np.corrcoef(basal_expr_pert, all_gene_expression[:, i])[0, 1]
                    if np.isnan(corr):
                        corr = 0
                    correlations[i] = corr
                pert_corrs[pert] = correlations

            with open(f"{self.data_path}/pert_corrs.pkl", "wb") as f:
                pkl.dump(pert_corrs, f)
        else:
            with open(f"{self.data_path}/pert_corrs.pkl", "rb") as f:
                pert_corrs = pkl.load(f)

        num_ctrl_cells = basal_ctrl_adata.shape[0]
        num_train_cells = train_target.shape[0]
        num_test_cells = test_target.shape[0]
        num_genes = basal_ctrl_adata.shape[1]

        pert_corr_train = np.zeros((num_train_cells, num_genes))
        for i, pert in tqdm(enumerate(all_perts_train), total=len(all_perts_train)):
            pert_corr_train[i, :] = pert_corrs[pert]

        pert_corr_test = np.zeros((num_test_cells, num_genes))
        for i, pert in tqdm(enumerate(all_perts_test), total=len(all_perts_test)):
            pert_corr_test[i, :] = pert_corrs[pert]

        random_train_mask = np.random.randint(0, num_ctrl_cells, num_train_cells)
        random_test_mask = np.random.randint(0, num_ctrl_cells, num_test_cells)

        train_input_expr = basal_ctrl_adata[random_train_mask, :].X.toarray()
        test_input_expr = basal_ctrl_adata[random_test_mask, :].X.toarray()

        raw_X_train = np.concatenate((train_input_expr, pert_corr_train), axis=1)
        raw_train_target = train_target.X.toarray()

        X_train, X_val, train_targets, val_targets = train_test_split(raw_X_train,
                                                                      raw_train_target,
                                                                      test_size=0.2)

        X_train = torch.from_numpy(X_train)
        train_target = torch.from_numpy(train_targets)
        X_val = torch.from_numpy(X_val)
        val_target = torch.from_numpy(val_targets)
        X_test = torch.from_numpy(np.concatenate((test_input_expr, pert_corr_test), axis=1))
        test_target = torch.from_numpy(test_target.X.toarray())

        with gzip.open(f"{self.data_path}/input_features/train_data_{self.spectral_parameter}.pkl.gz", "wb") as f:
            pkl.dump((X_train, train_target), f)
        with gzip.open(f"{self.data_path}/input_features/val_data_{self.spectral_parameter}.pkl.gz", "wb") as f:
            pkl.dump((X_val, val_target), f)
        with gzip.open(f"{self.data_path}/input_features/test_data_{self.spectral_parameter}.pkl.gz", "wb") as f:
            pkl.dump((X_test, test_target), f)

        return X_train, train_target, X_val, val_target, X_test, test_target

    def preprocess_replogle(self, adata):
        if not os.path.exists(f"{self.data_path}/{self.data_name}_filtered.h5ad"):
            adata.write(f"{self.data_path}/{self.data_name}_raw_counts.h5ad", compression='gzip')
            adata.layers["counts"] = adata.X.copy()
            sc.pp.normalize_total(adata)
            sc.pp.log1p(adata)
            adata.write(f"{self.data_path}/{self.data_name}_filtered.h5ad", compression='gzip')
        else:
            adata = sc.read_h5ad(f"{self.data_path}/{self.data_name}_filtered.h5ad")

        adata.obs['condition'] = adata.obs['perturbation'].replace('control', 'ctrl')

        genes = adata.var.index.to_list()
        genes_and_ctrl = genes + ['ctrl']

        # we remove the cells with perts that are not in the genes because we need gene expression values
        # to generate an in-silico perturbation embedding
        sg_pert_adata = adata[adata.obs['condition'].isin(genes_and_ctrl), :]

        ctrl_adata = sg_pert_adata[sg_pert_adata.obs['condition'] == 'ctrl', :]
        pert_adata = sg_pert_adata[sg_pert_adata.obs['condition'] != 'ctrl', :]
        all_perts = list(set(pert_adata.obs['condition'].to_list()))

        num_cells = ctrl_adata.shape[0]
        num_perts = len(all_perts)
        mask = np.zeros((num_cells, num_perts), dtype=bool)

        for i, pert in tqdm(enumerate(all_perts), total=len(all_perts)):
            pert_idx = genes.index(pert)
            non_zero_indices = ctrl_adata[:, pert_idx].X.sum(axis=1).nonzero()[0]
            num_non_zeroes = len(non_zero_indices)

            if len(non_zero_indices) < 500:
                sample_num = num_non_zeroes
            else:
                sample_num = 500

            sampled_indices = np.random.choice(non_zero_indices, sample_num, replace=False)

            mask[sampled_indices, i] = True

        mask_df = pd.DataFrame(mask, columns=all_perts)

        mask_df.to_pickle(f"{self.data_path}/{self.data_name}_mask_df.pkl")

        if not os.path.exists(f"{self.data_path}/all_perts.pkl"):
            with open(f"{self.data_path}/all_perts.pkl", "wb") as f:
                pkl.dump(all_perts, f)

        train, test, pert_list = get_splits.spectra(sg_pert_adata,
                                                    self.data_path,
                                                    self.spectra_params,
                                                    self.spectral_parameter
                                                    )
        return ctrl_adata, pert_adata, train, test, pert_list

    def featurise_replogle(self, pert_adata, pert_list, ctrl_adata, train, test):
        print(f"{self.data_name} dataset has {len(pert_list)} perturbations in common with the genes in the dataset.")

        all_perts = pert_adata.obs['condition'].to_list()

        if not os.path.exists(f"{self.data_path}/basal_ctrl_{self.data_name}_filtered.h5ad"):
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

        control_genes = basal_ctrl_adata.var.index.to_list()
        pert_genes = pert_adata.var.index.to_list()
        assert control_genes == pert_genes, ("Watch out! Genes in control and perturbation datasets are not the"
                                             " same, or are not indexed the same.")

        train_perts = [pert_list[i] for i in train]
        test_perts = [pert_list[i] for i in test]

        sc.pp.highly_variable_genes(pert_adata, n_top_genes=5000)
        highly_variable_genes = pert_adata.var_names[pert_adata.var['highly_variable']]
        hv_pert_adata = pert_adata[:, highly_variable_genes]

        train_target = hv_pert_adata[hv_pert_adata.obs['condition'].isin(train_perts), :]
        test_target = hv_pert_adata[hv_pert_adata.obs['condition'].isin(test_perts), :]

        all_perts_train = train_target.obs['condition'].values
        all_perts_test = test_target.obs['condition'].values

        perts_idx = {}
        for pert in all_perts:
            perts_idx[pert] = pert_genes.index(pert)

        num_ctrl_cells = basal_ctrl_adata.shape[0]
        num_train_cells = train_target.shape[0]
        num_test_cells = test_target.shape[0]
        num_genes = basal_ctrl_adata.shape[1]

        pass  # continue here

    def __getitem__(self, index):
        if self.stage == "train":
            return self.X_train[index], self.train_target[index]
        elif self.stage == "val":
            return self.X_val[index], self.val_target[index]
        elif self.eval_type is None:
            return self.X_test[index], self.test_target[index]
        else:
            assert "_de" in self.eval_type, "eval_type must be None or '{pert}_de'!"
            sp = self.spectral_parameter.split('_')[0]
            perturbed = self.eval_type.split('_')[0]
            with open(f"{self.data_path}/de_test/split_{sp}/{perturbed}_de_idx.pkl", "rb") as f:
                de_idx = pkl.load(f)
            return self.X_test[index], self.test_target[index], de_idx

    def __len__(self):
        if self.stage == "train":
            return len(self.X_train)
        elif self.stage == "val":
            return len(self.X_val)
        else:
            return len(self.X_test)
