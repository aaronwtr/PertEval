import torch
import os
import anndata
import gzip
import time

import numpy as np
import scanpy as sc
import pickle as pkl
import pandas as pd

from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from hydra.errors import HydraException
from scipy.stats import pearsonr

from src.utils.spectra import get_splits


class PerturbData(Dataset):
    def __init__(self, adata, data_path, spectral_parameter, spectra_params, fm, stage, **kwargs):
        self.data_name = data_path.split('/')[-1]
        self.data_path = data_path
        self.spectral_parameter = spectral_parameter
        self.spectra_params = spectra_params
        self.stage = stage
        self.fm = fm
        self.eval_type = kwargs.get("eval_type", None)

        if self.data_name == "norman":
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
            if not os.path.exists(f"{self.data_path}/input_features/train_data_{self.spectral_parameter}.pkl.gz"):
                ctrl_adata, pert_adata, train, test, pert_list = self.preprocess_replogle(adata)
                pp_data = self.featurise_replogle(pert_adata, pert_list, ctrl_adata, train, test)
                self.X_train, self.train_target, self.X_val, self.val_target, self.X_test, self.test_target = pp_data
            else:
                with gzip.open(f"{self.data_path}/input_features/train_data_{self.spectral_parameter}.pkl.gz", "rb") as f:
                    self.X_train, self.train_target = pkl.load(f)
                with gzip.open(f"{self.data_path}/input_features/val_data_{self.spectral_parameter}.pkl.gz", "rb") as f:
                    self.X_val, self.val_target = pkl.load(f)
                with gzip.open(f"{self.data_path}/input_features/test_data_{self.spectral_parameter}.pkl.gz", "rb") as f:
                    self.X_test, self.test_target = pkl.load(f)
        if self.data_name == "replogle_k562":
            if not os.path.exists(f"{self.data_path}/input_features/train_data_{self.spectral_parameter}.pkl.gz"):
                ctrl_adata, pert_adata, train, test, pert_list = self.preprocess_replogle(adata)
                pp_data = self.featurise_replogle(pert_adata, pert_list, ctrl_adata, train, test)
                self.X_train, self.train_target, self.X_val, self.val_target, self.X_test, self.test_target = pp_data
            else:
                with gzip.open(f"{self.data_path}/input_features/train_data_{self.spectral_parameter}.pkl.gz", "rb") as f:
                    self.X_train, self.train_target = pkl.load(f)
                with gzip.open(f"{self.data_path}/input_features/val_data_{self.spectral_parameter}.pkl.gz", "rb") as f:
                    self.X_val, self.val_target = pkl.load(f)
                with gzip.open(f"{self.data_path}/input_features/test_data_{self.spectral_parameter}.pkl.gz", "rb") as f:
                    self.X_test, self.test_target = pkl.load(f)

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

        # these just need to be the same between datasets, irrespective of order
        control_genes = basal_ctrl_adata.var.index.to_list()
        pert_genes = pert_adata.var.index.to_list()

        # these need to be paired between datasets, and in the same order
        pert_cell_conditions = pert_adata.obs['condition'].to_list()
        ctrl_cell_conditions = basal_ctrl_adata.obs['condition'].to_list()

        assert control_genes == pert_genes, ("Watch out! Genes in control and perturbation datasets are not the"
                                             " same, or are not indexed the same.")

        assert ctrl_cell_conditions == pert_cell_conditions, ("Watch out! Cell conditions in control and perturbation "
                                                              "datasets are not the or same, or are not indexed the "
                                                              "same!")

        train_perts = [pert_list[i] for i in train]
        test_perts = [pert_list[i] for i in test]

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

        with gzip.open(f"{self.data_path}/input_features/{self.fm}/train_data_{self.spectral_parameter}.pkl.gz", "wb") as f:
            pkl.dump((X_train, train_target), f)
        with gzip.open(f"{self.data_path}/input_features/{self.fm}/val_data_{self.spectral_parameter}.pkl.gz", "wb") as f:
            pkl.dump((X_val, val_target), f)
        with gzip.open(f"{self.data_path}/input_features/{self.fm}/test_data_{self.spectral_parameter}.pkl.gz", "wb") as f:
            pkl.dump((X_test, test_target), f)

        raise HydraException(f"Completed preprocessing and featurisation of split {self.spectral_parameter}. Moving "
                             f"on the next multirun...")

        return X_train, train_target, X_val, val_target, X_test, test_target

    def preprocess_replogle(self, adata):
        adata.obs['condition'] = adata.obs['perturbation'].replace('control', 'ctrl')

        genes = adata.var.index.to_list()
        genes_and_ctrl = genes + ['ctrl']

        # we remove the cells with perts that are not in the genes because we need gene expression values
        # to generate an in-silico perturbation embedding
        adata = adata[adata.obs['condition'].isin(genes_and_ctrl), :]
        unique_perts = list(set(adata.obs['condition'].to_list()))
        unique_perts.remove('ctrl')

        train, test, pert_list = get_splits.spectra(adata,
                                                    self.data_path,
                                                    self.spectra_params,
                                                    self.spectral_parameter
                                                    )

        sc.pp.normalize_total(adata)
        sc.pp.log1p(adata)
        sc.pp.highly_variable_genes(adata, n_top_genes=2000)
        highly_variable_genes = list(adata.var_names[adata.var['highly_variable']])
        missing_perts = list(set(unique_perts) - set(highly_variable_genes))
        combined_genes = list(set(highly_variable_genes) | set(missing_perts))
        adata = adata[:, combined_genes]

        ctrl_adata = adata[adata.obs['condition'] == 'ctrl', :]
        pert_adata = adata[adata.obs['condition'] != 'ctrl', :]

        num_cells = ctrl_adata.shape[0]
        num_perts = len(unique_perts)
        mask = np.zeros((num_cells, num_perts), dtype=bool)

        if not os.path.exists(f"{self.data_path}/{self.data_name}_mask_df.pkl"):
            for i, pert in tqdm(enumerate(unique_perts), total=len(unique_perts)):
                pert_idx = genes.index(pert)
                non_zero_indices = ctrl_adata[:, pert_idx].X.sum(axis=1).nonzero()[0]
                num_non_zeroes = len(non_zero_indices)

                if len(non_zero_indices) < 500:
                    sample_num = num_non_zeroes
                else:
                    sample_num = 500

                sampled_indices = np.random.choice(non_zero_indices, sample_num, replace=False)

                mask[sampled_indices, i] = True

            mask_df = pd.DataFrame(mask, columns=unique_perts)

            mask_df.to_pickle(f"{self.data_path}/{self.data_name}_mask_df.pkl")

        if not os.path.exists(f"{self.data_path}/all_perts.pkl"):
            with open(f"{self.data_path}/all_perts.pkl", "wb") as f:
                pkl.dump(unique_perts, f)

        return ctrl_adata, pert_adata, train, test, pert_list

    def featurise_replogle(self, pert_adata, pert_list, ctrl_adata, train, test):
        print(f"{self.data_name} dataset has {len(pert_list)} perturbations in common with the genes in the dataset.")

        if not os.path.exists(f"{self.data_path}/basal_ctrl_{self.data_name}_filtered.h5ad"):
            ctrl_X = ctrl_adata.X.toarray()
            basal_ctrl_X = np.zeros((pert_adata.shape[0], ctrl_X.shape[1]))
            subset_size = 500

            for cell in tqdm(range(pert_adata.shape[0])):
                subset = ctrl_X[np.random.choice(ctrl_X.shape[0], subset_size), :]
                basal_ctrl_X[cell, :] = subset.mean(axis=0)

            # we add pert_adata to obs because we want to pair control expression to perturbed cells
            basal_ctrl_adata = anndata.AnnData(X=basal_ctrl_X, obs=pert_adata.obs, var=ctrl_adata.var)

            # noinspection PyTypeChecker
            basal_ctrl_adata.write(f"{self.data_path}/basal_ctrl_{self.data_name}_filtered.h5ad")
        else:
            basal_ctrl_adata = sc.read_h5ad(f"{self.data_path}/basal_ctrl_{self.data_name}_filtered.h5ad")

        # these just need to be the same between datasets, irrespective of order
        control_genes = sorted(basal_ctrl_adata.var_names.to_list())
        pert_genes = sorted(pert_adata.var_names.to_list())

        # these need to be paired between datasets, and in the same order
        pert_cell_conditions = pert_adata.obs['condition'].to_list()
        ctrl_cell_conditions = basal_ctrl_adata.obs['condition'].to_list()

        assert control_genes == pert_genes, "Watch out! Genes in control and perturbation datasets do not match!"

        assert ctrl_cell_conditions == pert_cell_conditions, ("Watch out! Cell conditions in control and perturbation "
                                                              "datasets are not the or same, or are not indexed the "
                                                              "same!")

        train_perts = [pert_list[i] for i in train]
        test_perts = [pert_list[i] for i in test]

        train_target = pert_adata[pert_adata.obs['condition'].isin(train_perts), :]
        test_target = pert_adata[pert_adata.obs['condition'].isin(test_perts), :]

        all_perts_train = train_target.obs['condition'].values
        all_perts_test = test_target.obs['condition'].values

        if not os.path.exists(f"{self.data_path}/pert_corrs.pkl.gz"):
            all_gene_expression = basal_ctrl_adata.X

            results = []
            for pert in tqdm(pert_list, total=len(pert_list)):
                pert, correlations = self.compute_correlations(pert, basal_ctrl_adata, all_gene_expression)
                results.append((pert, correlations))

            pert_corrs = {pert: corr for pert, corr in results}

            with gzip.open(f"{self.data_path}/pert_corrs.pkl.gz", "wb") as f:
                pkl.dump(pert_corrs, f)
        else:
            with gzip.open(f"{self.data_path}/pert_corrs.pkl.gz", "rb") as f:
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

        print("\n\nPertubation correlation features computed.\n\n")

        random_train_mask = np.random.randint(0, num_ctrl_cells, num_train_cells)
        random_test_mask = np.random.randint(0, num_ctrl_cells, num_test_cells)

        # if not os.path.exists(f"{self.data_path}/random_train_mask.pkl.gz"):
        #     for random_train_chunk in self.generate_random_in_chunks(0, num_ctrl_cells, num_train_cells):
        #         if 'random_train_mask' not in locals():
        #             random_train_mask = random_train_chunk
        #         else:
        #             random_train_mask = np.concatenate((random_train_mask, random_train_chunk))
        #     with gzip.open(f"{self.data_path}/random_train_mask.pkl.gz", "wb") as f:
        #         pkl.dump(random_train_mask, f)
        # else:
        #     with gzip.open(f"{self.data_path}/random_train_mask.pkl.gz", "rb") as f:
        #         random_train_mask = pkl.load(f)
        #
        # if not os.path.exists(f"{self.data_path}/random_test_mask.pkl.gz"):
        #     for random_test_chunk in self.generate_random_in_chunks(0, num_ctrl_cells, num_test_cells):
        #         if 'random_test_mask' not in locals():
        #             random_test_mask = random_test_chunk
        #         else:
        #             random_test_mask = np.concatenate((random_test_mask, random_test_chunk))
        #     with gzip.open(f"{self.data_path}/random_test_mask.pkl.gz", "wb") as f:
        #         pkl.dump(random_test_mask, f)
        # else:
        #     with gzip.open(f"{self.data_path}/random_test_mask.pkl.gz", "rb") as f:
        #         random_test_mask = pkl.load(f)

        print("\n\nInput masks generated.\n\n")

        time_elapsed = time.time() - self.start_time
        print(f"\n\nTime elapsed: {time.strftime('%H:%M:%S', time.gmtime(time_elapsed))}\n\n")

        train_input_expr = basal_ctrl_adata[random_train_mask, :].X.toarray()
        test_input_expr = basal_ctrl_adata[random_test_mask, :].X.toarray()

        print("\n\nInput expression data generated.\n\n")

        print("Train target shape:", train_target.shape)
        print("\nTrain input expression shape and pert_corr shape\n")
        print(train_input_expr.shape, pert_corr_train.shape)
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

        print("\n\nData split into train, validation, and test sets.\n\n")
        time_elapsed = time.time() - self.start_time
        print(f"\nnTime elapsed: {time.strftime('%H:%M:%S', time.gmtime(time_elapsed))}\nn")

        # save data as pickle without gzip

        with open(f"{self.data_path}/input_features/train_data_{self.spectral_parameter}.pkl", "wb") as f:
            pkl.dump((X_train, train_target), f)
        with open(f"{self.data_path}/input_features/val_data_{self.spectral_parameter}.pkl", "wb") as f:
            pkl.dump((X_val, val_target), f)
        with open(f"{self.data_path}/input_features/test_data_{self.spectral_parameter}.pkl", "wb") as f:
            pkl.dump((X_test, test_target), f)

        # with gzip.open(f"{self.data_path}/input_features/train_data_{self.spectral_parameter}.pkl.gz", "wb") as f:
        #     pkl.dump((X_train, train_target), f)
        #
        # with gzip.open(f"{self.data_path}/input_features/val_data_{self.spectral_parameter}.pkl.gz", "wb") as f:
        #     pkl.dump((X_val, val_target), f)
        # with gzip.open(f"{self.data_path}/input_features/test_data_{self.spectral_parameter}.pkl.gz", "wb") as f:
        #     pkl.dump((X_test, test_target), f)

        print("\n\nData saved to disk.\n\n")
        time_elapsed = time.time() - self.start_time
        print(f"\nnTime elapsed: {time.strftime('%H:%M:%S', time.gmtime(time_elapsed))}\nn")

        raise HydraException(f"Completed preprocessing and featurisation of split {self.spectral_parameter}. Moving "
                             f"on the next multirun...")

        return X_train, train_target, X_val, val_target, X_test, test_target

    @staticmethod
    def compute_correlations(pert, basal_ctrl_adata, all_gene_expression):
        pert_idx = basal_ctrl_adata.var_names.get_loc(pert)
        basal_expr_pert = basal_ctrl_adata.X[:, pert_idx].flatten()
        correlations = np.array(
            [pearsonr(basal_expr_pert, all_gene_expression[:, i])[0] for i in range(all_gene_expression.shape[1])])
        correlations[np.isnan(correlations)] = 0
        return pert, correlations

    @staticmethod
    def generate_random_in_chunks(low, high, num_total, chunk_size=1000):
        num_generated = 0
        pbar = tqdm(total=num_total)
        while num_generated < num_total:
            num_to_generate = min(chunk_size, num_total - num_generated)
            yield np.random.randint(low, high, num_to_generate)
            num_generated += num_to_generate
            pbar.update(num_to_generate)
        pbar.close()

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
