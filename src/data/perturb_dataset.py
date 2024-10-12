import torch
import os
from pathlib import PurePath
import anndata
import gzip
import gdown

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
        self.data_name = PurePath(data_path).parts[-1]
        self.data_path = data_path
        self.spectral_parameter = spectral_parameter
        self.spectra_params = spectra_params
        self.stage = stage
        self.fm = fm
        self.data_processor = None
        self.deg_dict = None
        self.basal_ctrl_adata = None
        self.genes = None
        self.all_perts_train = None
        self.all_perts_test = None

        if kwargs:
            if 'deg_dict' in kwargs and 'perturbation' in kwargs:
                self.deg_dict = kwargs['deg_dict']
                self.perturbation = kwargs['perturbation']
            else:
                raise HydraException("kwargs can only contain 'perturbation' and 'deg_dict' keys!")

        if self.fm == 'mean':
            # use raw_expression data to calculate mean expression
            self.fm = 'raw_expression'

        assert self.fm in ["raw_expression", "scgpt", "geneformer", "scfoundation", "scbert", "uce"], \
            "fm must be set to 'raw_expression', 'scgpt', 'geneformer', 'scfoundation', 'scbert', or 'uce'!"

        feature_path = f"{self.data_path}/input_features/{self.fm}"

        if not os.path.exists(feature_path):
            os.makedirs(feature_path)

        if self.data_name == "norman_1":
            if not os.path.exists(f"{feature_path}/train_data_{self.spectral_parameter}.pkl.gz"):
                (self.X_train, self.train_target, self.X_val, self.val_target, self.X_test, self.test_target,
                 self.ctrl_expr, _) = self.preprocess_and_featurise_norman(adata)
            else:
                self.basal_ctrl_adata = sc.read_h5ad(f"{self.data_path}/basal_ctrl_{self.data_name}_pp_filtered.h5ad")
                with gzip.open(f"{feature_path}/train_data_{self.spectral_parameter}.pkl.gz", "rb") as f:
                    self.X_train, self.train_target = pkl.load(f)
                with gzip.open(f"{feature_path}/val_data_{self.spectral_parameter}.pkl.gz", "rb") as f:
                    self.X_val, self.val_target = pkl.load(f)
                with gzip.open(f"{feature_path}/test_data_{self.spectral_parameter}.pkl.gz", "rb") as f:
                    self.X_test, self.test_target = pkl.load(f)
                with open(f"{self.data_path}/raw_expression_{self.data_name}_pp_filtered.pkl", "rb") as f:
                    self.ctrl_expr = pkl.load(f)

        if self.data_name == "norman_2":
            if not os.path.exists(f"{feature_path}/train_data_{self.spectral_parameter}.pkl.gz"):
                (self.X_train, self.train_target, self.X_val, self.val_target, self.X_test, self.test_target,
                 self.ctrl_expr, self.all_perts_test) = self.preprocess_and_featurise_norman(adata)
            else:
                self.basal_ctrl_adata = sc.read_h5ad(f"{self.data_path}/basal_ctrl_{self.data_name}_pp_filtered.h5ad")
                with gzip.open(f"{feature_path}/train_data_{self.spectral_parameter}.pkl.gz", "rb") as f:
                    self.X_train, self.train_target = pkl.load(f)
                with gzip.open(f"{feature_path}/val_data_{self.spectral_parameter}.pkl.gz", "rb") as f:
                    self.X_val, self.val_target = pkl.load(f)
                with gzip.open(f"{feature_path}/test_data_{self.spectral_parameter}.pkl.gz", "rb") as f:
                    self.X_test, self.test_target = pkl.load(f)
                with open(f"{self.data_path}/raw_expression_{self.data_name}_pp_filtered.pkl", "rb") as f:
                    self.ctrl_expr = pkl.load(f)
                with open(f"{self.data_path}/target_perts/all_perts_test_{self.spectral_parameter}.pkl", "rb") as f:
                    self.all_perts_test = pkl.load(f)

        if self.data_name == "replogle_rpe1":
            if not os.path.exists(f"{self.data_path}/input_features/train_data_{self.spectral_parameter}.pkl.gz"):
                ctrl_adata, pert_adata, train, test, pert_list = self.preprocess_replogle(adata)
                pp_data = self.featurise_replogle(pert_adata, pert_list, ctrl_adata, train, test)
                self.X_train, self.train_target, self.X_val, self.val_target, self.X_test, self.test_target = pp_data
            else:
                with gzip.open(f"{self.data_path}/input_features/train_data_{self.spectral_parameter}.pkl.gz",
                               "rb") as f:
                    self.X_train, self.train_target = pkl.load(f)
                with gzip.open(f"{self.data_path}/input_features/val_data_{self.spectral_parameter}.pkl.gz", "rb") as f:
                    self.X_val, self.val_target = pkl.load(f)
                with gzip.open(f"{self.data_path}/input_features/test_data_{self.spectral_parameter}.pkl.gz",
                               "rb") as f:
                    self.X_test, self.test_target = pkl.load(f)

        if self.data_name == "replogle_k562":
            if not os.path.exists(f"{self.data_path}/input_features/train_data_{self.spectral_parameter}.pkl.gz"):
                ctrl_adata, pert_adata, train, test, pert_list = self.preprocess_replogle(adata)
                pp_data = self.featurise_replogle(pert_adata, pert_list, ctrl_adata, train, test)
                self.X_train, self.train_target, self.X_val, self.val_target, self.X_test, self.test_target = pp_data
            else:
                with gzip.open(f"{self.data_path}/input_features/train_data_{self.spectral_parameter}.pkl.gz",
                               "rb") as f:
                    self.X_train, self.train_target = pkl.load(f)
                with gzip.open(f"{self.data_path}/input_features/val_data_{self.spectral_parameter}.pkl.gz", "rb") as f:
                    self.X_val, self.val_target = pkl.load(f)
                with gzip.open(f"{self.data_path}/input_features/test_data_{self.spectral_parameter}.pkl.gz",
                               "rb") as f:
                    self.X_test, self.test_target = pkl.load(f)

    def preprocess_and_featurise_norman(self, adata):
        nonzero_genes = (adata.X.sum(axis=0) > 5).A1
        filtered_adata = adata[:, nonzero_genes]
        adata.obs['condition'] = adata.obs['guide_ids'].cat.rename_categories({'': 'ctrl'})
        adata.obs['guide_ids'] = adata.obs['guide_ids'].cat.remove_categories('')

        if self.data_name == "norman_1":
            single_gene_mask = [True if "," not in name else False for name in adata.obs['condition']]
            adata = filtered_adata[single_gene_mask, :]
        else:
            adata.obs['condition'] = adata.obs['condition'].str.replace(',', '+')

        self.genes = adata.var['gene_symbols'].to_list()
        genes_and_ctrl = self.genes + ['ctrl']

        # we remove the cells with perts that are not in the genes because we need gene expression values
        # to generate an in-silico perturbation embedding
        if self.data_name == "norman_1":
            adata = adata[adata.obs['condition'].isin(genes_and_ctrl), :]
        else:
            conditions = adata.obs['condition']

            # need to account for the two-gene perturbations
            filtered_conditions = conditions.apply(
                lambda cond: cond in genes_and_ctrl or (
                        '+' in cond and all(gene in genes_and_ctrl for gene in cond.split('+'))
                )
            )
            adata = adata[filtered_conditions, :]

        train, test, pert_list = get_splits.spectra(adata,
                                                    self.data_path,
                                                    self.spectra_params,
                                                    self.spectral_parameter
                                                    )

        print(f"Norman dataset has {len(pert_list)} perturbations in common with the genes in the dataset.")

        ctrl_adata = adata[adata.obs['condition'] == 'ctrl', :]

        pert_adata = adata[adata.obs['condition'] != 'ctrl', :]
        all_perts = list(set(pert_adata.obs['condition'].to_list()))

        num_cells = ctrl_adata.shape[0]
        num_perts = len(all_perts)

        # generate embedding mask for the perturbable genes with nonzero expression values
        if self.data_name == "norman_1":
            if not os.path.exists(f"{self.data_path}/norman_mask_df.pkl"):
                mask = np.zeros((num_cells, num_perts), dtype=bool)

                for idx, pert in enumerate(all_perts):
                    mask = self.sg_pert_mask(mask, pert, idx, ctrl_adata)

                mask_df = pd.DataFrame(mask, columns=all_perts)
                mask_df.to_pickle(f"{self.data_path}/norman_mask_df.pkl")
            else:
                mask_df = pd.read_pickle(f"{self.data_path}/norman_mask_df.pkl")
        else:
            if not os.path.exists(f"{self.data_path}/norman_mask_dg_df.pkl"):
                mask = np.zeros((num_cells, num_perts), dtype=bool)

                for idx, pert in enumerate(all_perts):
                    if '+' not in pert:
                        mask = self.sg_pert_mask(mask, pert, idx, ctrl_adata)
                    else:
                        pert1, pert2 = pert.split('+')
                        try:
                            pert_idx_1 = self.genes.index(pert1)
                            pert_idx_2 = self.genes.index(pert2)
                        except ValueError:
                            print(f"{pert} not found in the gene list. Cannot do in silico perturbation.")
                            continue

                        # Find indices where both pert1 and pert2 are non-zero
                        both_non_zero_indices = np.intersect1d(
                            ctrl_adata[:, pert_idx_1].X.nonzero()[0],
                            ctrl_adata[:, pert_idx_2].X.nonzero()[0]
                        )

                        # Find indices where either pert1 or pert2 is non-zero
                        either_non_zero_indices = np.union1d(
                            ctrl_adata[:, pert_idx_1].X.nonzero()[0],
                            ctrl_adata[:, pert_idx_2].X.nonzero()[0]
                        )

                        # Sample cells
                        sampled_indices = []
                        if len(both_non_zero_indices) > 0:
                            sampled_indices.extend(both_non_zero_indices)

                        if len(sampled_indices) < 500:
                            remaining_sample_size = 500 - len(sampled_indices)
                            additional_indices = np.setdiff1d(either_non_zero_indices, both_non_zero_indices)
                            if len(additional_indices) > 0:
                                sampled_indices.extend(np.random.choice(additional_indices, min(remaining_sample_size,
                                                                                                len(additional_indices)),
                                                                        replace=False))

                        sampled_indices = np.array(sampled_indices[:500])
                        mask[sampled_indices, idx] = True

                mask_df = pd.DataFrame(mask, columns=all_perts)
                mask_df.to_pickle(f"{self.data_path}/norman_mask_df_dg.pkl")
            else:
                mask_df = pd.read_pickle(f"{self.data_path}/norman_mask_df_dg.pkl")

        mask_df_cells = mask_df.any(axis=0)
        unique_perts = list(mask_df.columns[mask_df_cells])

        gene_to_ensg = dict(zip(adata.var['gene_symbols'], adata.var_names))

        if self.fm != 'raw_expression':

            # dictionary lookup of the file id for embeddings download
            embeddings_file_id = {
                'geneformer': {
                                'norman_1': {'ctrl': "1yyQRcZEhdcsLOeMQKjZp7eSq6uHIYpwc", 'pert': "1nK-YeenYax84vV1LVLghg2aXF92CvvuA"},
                                'norman_2': {'ctrl': "1DTuCqigOr4lIg8J2qBFnUIHXz3oMQQC1", 'pert': "1VWiv7VOaqra63HFocNWkWd-u-MxvGdfY"}},
                'scgpt': {
                                'norman_1': {'ctrl': '1ECHWwA5idPQspwfS74PMunxtN9Uj0CzR', 'pert': '1T1Vd779feygiDhW1zmWI67uN8dDQJcnq'},
                                'norman_2': {'ctrl': '1Oy9u-YxyoQGjYLEclKyrSOf8G4LkrcKk', 'pert': '1v2wH3pr9TcSrTceRfFG_KBDsLg7RfTiR'}},
                'scfoundation': {
                                'norman_1': {'ctrl': '1JVLfShRXjwUgovX78qWDMiL1ZXhVgepQ', 'pert': '1CeYuSuUP408h33o11L1e82chV9WVwZXY'},
                                'norman_2': {'ctrl': '1VHEV-lgPb2xe362yM3h6_NLhSBkn1jtp', 'pert': '1KaWNIJe--NPj5u7k00D0CCrTRJuunu98'}},
                'scbert': {
                                'norman_1': {'ctrl': '15p0kvoImPNfl31qYmTtGuyhB4GsaSbeu', 'pert': '1S1lMR6UhM5QUik0imv8A7G5f3i4xFj7R'},
                                'norman_2': {'ctrl': '10JD689TmbvRAGzsQ9vwwHyvMsb4kv30f', 'pert': '1wl5GXnXbCU7ACtO4YF3Ii0kFVsIJhyOr'}},
                'uce': {
                                'norman_1': {'ctrl': '1AbhxtmJ8aYlMqprQC_h104elEGC3sBXX', 'pert': '1R3mRXLSwnlnKgSNBYEKQ5THAOqhfS9u5'},
                                'norman_2': {'ctrl': '1yixfxWSMSzdjotBfrrwsHHqUuKCSJZ15', 'pert': '1FQvKSFma0TGO2RCOkOSWYLNREeUOjAKi'}}
                                  }

            # create embeddings folder if it does not exist
            if not os.path.exists(f"{self.data_path}/embeddings"):
                os.makedirs(f"{self.data_path}/embeddings", exist_ok=True)

            # check the embeddings have been downloaded for the scFMs
            if not os.path.exists(f"{self.data_path}/embeddings/{self.data_name}_{self.fm}_fm_ctrl.pkl.gz"):
                print(f"Downloading embeddings for {self.data_name} {self.fm} control data...")
                # get file ID
                file_id = embeddings_file_id[self.fm][self.data_name]['ctrl']
                filename = f"{self.data_name}_{self.fm}_fm_ctrl.pkl.gz"
                gdown.download(id=file_id,
                               output=f"{self.data_path}/embeddings/{filename}")
            if not os.path.exists(f"{self.data_path}/embeddings/{self.data_name}_{self.fm}_fm_pert.pkl.gz"):
                print(f"Downloading embeddings for {self.data_name} {self.fm} pertubation data...")
                # get file ID
                file_id = embeddings_file_id[self.fm][self.data_name]['pert']
                filename = f"{self.data_name}_{self.fm}_fm_pert.pkl.gz"
                gdown.download(id=file_id,
                                 output=f"{self.data_path}/embeddings/{filename}")
            # load the embeddings
            with gzip.open(f"{self.data_path}/embeddings/{self.data_name}_{self.fm}_fm_ctrl.pkl.gz","rb") as f:
                fm_ctrl_data = pkl.load(f)
            with gzip.open(f"{self.data_path}/embeddings/{self.data_name}_{self.fm}_fm_pert.pkl.gz", "rb") as f:
                fm_pert_data = pkl.load(f)

            fm_pert_data = {pert: emb for pert, emb in fm_pert_data.items() if emb.shape[0] > 0}

            assert isinstance(fm_ctrl_data, (np.ndarray, anndata.AnnData, pd.DataFrame)), ("fm_ctrl_data should be an "
                                                                                           "array, dataframe or h5ad "
                                                                                           "file!")

            if isinstance(fm_ctrl_data, anndata.AnnData):
                assert hasattr(fm_ctrl_data, 'obsm'), "fm_ctrl_data should have an attribute 'obsm'!"
                fm_ctrl_X = fm_ctrl_data.obsm['X']
            elif isinstance(fm_ctrl_data, pd.DataFrame):
                fm_ctrl_X = fm_ctrl_data.values
            else:
                fm_ctrl_X = fm_ctrl_data

            assert isinstance(fm_pert_data, dict), ("fm_pert_data should be a dictionary with perturbed gene as key and"
                                                    "embedding as value!")

        basal_ctrl_path = f"{self.data_path}/basal_ctrl_{self.data_name}_pp_filtered.h5ad"

        # The reason this needs to be regenerated for each model, is that the embedding dimensions are different
        # for each model
        embed_basal_ctrl_path = f"{self.data_path}/embed_basal_ctrl_{self.data_name}_{self.fm}_pp_filtered.h5ad"

        basal_ctrl_not_exists = not os.path.exists(basal_ctrl_path)
        embed_basal_ctrl_condition = (os.path.exists(basal_ctrl_path) and not os.path.exists(embed_basal_ctrl_path)
                                      and self.fm != 'raw_expression')

        if basal_ctrl_not_exists or embed_basal_ctrl_condition:
            # Condensed the logic, but it is saying that if the basal_ctrl_adata does not exist, or if it does exists
            # but the embed_basal_ctrl_adata does not exist, then we need to regenerate the basal_ctrl_adata for the
            # scFM model

            # Save control_data_raw for inference with scFMs and pert_data for contextual alignment experiment
            if not os.path.exists(f"{self.data_path}/ctrl_{self.data_name}_raw_counts.h5ad"):
                ctrl_adata.write(f"{self.data_path}/ctrl_{self.data_name}_raw_counts.h5ad", compression='gzip')
            if not os.path.exists(f"{self.data_path}/pert_{self.data_name}_raw_counts.h5ad"):
                pert_adata.write(f"{self.data_path}/pert_{self.data_name}_raw_counts.h5ad", compression='gzip')

            if not os.path.exists(f"{self.data_path}/{self.data_name}_pp_ctrl_filtered.h5ad"):
                # This is the same between all models
                sc.pp.normalize_total(adata)
                sc.pp.log1p(adata)
                sc.pp.highly_variable_genes(adata, n_top_genes=2000)
                highly_variable_genes = pert_adata.var_names[adata.var['highly_variable']]
                if self.data_name == "norman_1":
                    unique_perts_ensg = [gene_to_ensg[pert] for pert in unique_perts]
                else:
                    unique_perts_ensg = [gene_to_ensg[pert] for pert in unique_perts if '+' not in pert]
                missing_perts = list(set(unique_perts_ensg) - set(highly_variable_genes))
                combined_genes = list(set(highly_variable_genes) | set(missing_perts))
                hvg_adata = adata[:, combined_genes]

                pert_adata = hvg_adata[hvg_adata.obs['condition'] != 'ctrl', :]
                pert_adata = pert_adata[pert_adata.obs['condition'].isin(unique_perts), :]

                pert_adata.write(f"{self.data_path}/{self.data_name}_pp_pert_filtered.h5ad", compression='gzip')

                ctrl_adata = hvg_adata[hvg_adata.obs['condition'] == 'ctrl', :]
                ctrl_adata.write(f"{self.data_path}/{self.data_name}_pp_ctrl_filtered.h5ad", compression='gzip')
            else:
                ctrl_adata = sc.read_h5ad(f"{self.data_path}/{self.data_name}_pp_ctrl_filtered.h5ad")
                pert_adata = sc.read_h5ad(f"{self.data_path}/{self.data_name}_pp_pert_filtered.h5ad")

            subset_size = 500
            if basal_ctrl_not_exists:
                # equal subsampling to pair control cells with perturbed cells
                ctrl_X = ctrl_adata.X.toarray()

                basal_ctrl_X = np.zeros((pert_adata.shape[0], ctrl_X.shape[1]))
                for cell in tqdm(range(pert_adata.shape[0])):
                    random_cells = np.random.choice(ctrl_X.shape[0], subset_size)
                    subset_X = ctrl_X[random_cells, :]
                    basal_ctrl_X[cell, :] = subset_X.mean(axis=0)
                basal_ctrl_adata = anndata.AnnData(X=basal_ctrl_X, obs=pert_adata.obs, var=ctrl_adata.var)

                # noinspection PyTypeChecker
                basal_ctrl_adata.write(basal_ctrl_path, compression='gzip')
            else:
                basal_ctrl_adata = sc.read_h5ad(basal_ctrl_path)

            if not os.path.exists(f"{self.data_path}/raw_expression_{self.data_name}_pp_filtered.pkl"):
                ctrl_expr = basal_ctrl_adata[basal_ctrl_adata.obs['condition'].isin(unique_perts), :]
                ctrl_expr = ctrl_expr.X.toarray()
                with open(f"{self.data_path}/raw_expression_{self.data_name}_pp_filtered.pkl", "wb") as f:
                    pkl.dump(ctrl_expr, f)
            else:
                with open(f"{self.data_path}/raw_expression_{self.data_name}_pp_filtered.pkl", "rb") as f:
                    ctrl_expr = pkl.load(f)

            if self.fm != 'raw_expression':
                basal_ctrl_X = np.zeros((pert_adata.shape[0], fm_ctrl_X.shape[1]))
                for cell in tqdm(range(pert_adata.shape[0])):
                    random_cells = np.random.choice(fm_ctrl_X.shape[0], subset_size)
                    subset_X = fm_ctrl_X[random_cells, :]
                    basal_ctrl_X[cell, :] = subset_X.mean(axis=0)

                basal_ctrl_X_empty = np.zeros((pert_adata.shape[0], fm_ctrl_X.shape[1]))
                basal_ctrl_adata = anndata.AnnData(X=basal_ctrl_X_empty, obs=pert_adata.obs)
                basal_ctrl_adata.obsm['X'] = basal_ctrl_X

                basal_ctrl_adata.write(embed_basal_ctrl_path, compression='gzip')
        else:
            if self.fm == 'raw_expression':
                with open(f"{self.data_path}/raw_expression_{self.data_name}_pp_filtered.pkl", "rb") as f:
                    ctrl_expr = pkl.load(f)
                basal_ctrl_adata = sc.read_h5ad(f"{self.data_path}/basal_ctrl_{self.data_name}_pp_filtered.h5ad")
                pert_adata = sc.read_h5ad(f"{self.data_path}/{self.data_name}_pp_pert_filtered.h5ad")
            else:
                with open(f"{self.data_path}/raw_expression_{self.data_name}_pp_filtered.pkl", "rb") as f:
                    ctrl_expr = pkl.load(f)
                basal_ctrl_adata = sc.read_h5ad(embed_basal_ctrl_path)
                pert_adata = sc.read_h5ad(f"{self.data_path}/{self.data_name}_pp_pert_filtered.h5ad")
                if self.data_name == "norman_1":
                    emb_perts = fm_pert_data.keys()
                    pert_adata = pert_adata[pert_adata.obs['condition'].isin(emb_perts), :]

        pert_cell_conditions = pert_adata.obs['condition'].to_list()
        ctrl_cell_conditions = basal_ctrl_adata.obs['condition'].to_list()

        assert ctrl_cell_conditions == pert_cell_conditions, ("Watch out! Cell conditions in control and perturbation "
                                                              "datasets are not the same, or are not indexed the "
                                                              "same!")
        train_perts = [pert_list[i] for i in train]
        test_perts = [pert_list[i] for i in test]

        train_target = pert_adata[pert_adata.obs['condition'].isin(train_perts), :]
        test_target = pert_adata[pert_adata.obs['condition'].isin(test_perts), :]

        self.all_perts_train = train_target.obs['condition'].values
        self.all_perts_test = test_target.obs['condition'].values

        # check if there exists a target_perts folder yet, if not make it
        if not os.path.exists(f"{self.data_path}/target_perts"):
            os.makedirs(f"{self.data_path}/target_perts")

        with open(f"{self.data_path}/target_perts/all_perts_test_{self.spectral_parameter}.pkl", "wb") as f:
            pkl.dump(self.all_perts_test, f)

        unique_perts = list(set(basal_ctrl_adata.obs['condition'].to_list()))

        if self.fm == 'raw_expression':
            if not os.path.exists(f"{self.data_path}/pert_corrs.pkl"):
                all_gene_expression = basal_ctrl_adata.X

                processed_perts = []
                pert_corrs = {}
                for pert in tqdm(unique_perts, total=len(unique_perts)):
                    correlations = np.zeros(basal_ctrl_adata.shape[1])
                    if pert in processed_perts:
                        continue
                    if '+' in pert:
                        for _pert in pert.split('+'):
                            ensg_id = gene_to_ensg[_pert]
                            pert_idx = basal_ctrl_adata.var_names.get_loc(ensg_id)
                            basal_expr_pert = basal_ctrl_adata.X[:, pert_idx].flatten()
                            for i in range(all_gene_expression.shape[1]):
                                corr = np.corrcoef(basal_expr_pert, all_gene_expression[:, i])[0, 1]
                                if np.isnan(corr):
                                    corr = 0
                                correlations[i] = corr
                            processed_perts.append(_pert)
                            pert_corrs[_pert] = correlations
                    else:
                        ensg_id = gene_to_ensg[pert]
                        pert_idx = basal_ctrl_adata.var_names.get_loc(ensg_id)
                        basal_expr_pert = basal_ctrl_adata.X[:, pert_idx].flatten()
                        for i in range(all_gene_expression.shape[1]):
                            corr = np.corrcoef(basal_expr_pert, all_gene_expression[:, i])[0, 1]
                            if np.isnan(corr):
                                corr = 0
                            correlations[i] = corr
                        processed_perts.append(pert)
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

        random_train_mask = np.random.randint(0, num_ctrl_cells, num_train_cells)
        random_test_mask = np.random.randint(0, num_ctrl_cells, num_test_cells)

        if self.fm == "raw_expression":
            pert_corr_train = np.zeros((num_train_cells, num_genes))
            for i, pert in tqdm(enumerate(self.all_perts_train), total=len(self.all_perts_train)):
                if '+' in pert:
                    for _pert in pert.split('+'):
                        pert_corr_train[i, :] += pert_corrs[_pert]
                    pert_corr_train[i, :] /= len(pert.split('+'))
                else:
                    pert_corr_train[i, :] = pert_corrs[pert]

            pert_corr_test = np.zeros((num_test_cells, num_genes))
            for i, pert in tqdm(enumerate(self.all_perts_test), total=len(self.all_perts_test)):
                if '+' in pert:
                    for _pert in pert.split('+'):
                        pert_corr_test[i, :] += pert_corrs[_pert]
                    pert_corr_test[i, :] /= len(pert.split('+'))
                else:
                    pert_corr_test[i, :] = pert_corrs[pert]

            train_input_expr = basal_ctrl_adata[random_train_mask, :].X.toarray()
            test_input_expr = basal_ctrl_adata[random_test_mask, :].X.toarray()

            raw_X_train = np.concatenate((train_input_expr, pert_corr_train), axis=1)
            X_test = np.concatenate((test_input_expr, pert_corr_test), axis=1)
        else:
            if isinstance(basal_ctrl_adata, anndata.AnnData):
                obsm_keys = basal_ctrl_adata.obsm.keys()
                for key in obsm_keys:
                    if 'X' in key:
                        ctrl_embs = basal_ctrl_adata.obsm[key]
                        train_input_emb = ctrl_embs[random_train_mask, :]
                        test_input_emb = ctrl_embs[random_test_mask, :]
                        break
                    else:
                        raise KeyError("basal_ctrl_adata should have an attribute 'obsm' with 'X' key!")
                else:
                    raise KeyError("basal_ctrl_adata should be AnnData with 'obsm' attribute with 'X' key!")

            emb_dim = fm_ctrl_X.shape[1]
            pert_embs_train = np.zeros((num_train_cells, emb_dim))
            if self.data_name == "norman_1":
                for i, pert in enumerate(self.all_perts_train):
                    pert_embs_train[i, :] = fm_pert_data[pert].mean(axis=0)

                pert_embs_test = np.zeros((num_test_cells, emb_dim))
                for i, pert in enumerate(self.all_perts_test):
                    pert_embs_test[i, :] = fm_pert_data[pert].mean(axis=0)
            else:
                for i, pert in enumerate(self.all_perts_train):
                    # Only consider 2-gene perturbations
                    if '+' in pert:
                        pert_embs_train[i, :] = fm_pert_data[pert].mean(axis=0)
                pert_embs_test = np.zeros((num_test_cells, emb_dim))
                for i, pert in enumerate(self.all_perts_test):
                    if '+' in pert:
                        pert_embs_test[i, :] = fm_pert_data[pert].mean(axis=0)

            raw_X_train = np.concatenate((train_input_emb, pert_embs_train), axis=1)
            X_test = np.concatenate((test_input_emb, pert_embs_test), axis=1)

        raw_train_target = train_target.X.toarray()

        X_train, X_val, train_targets, val_targets = train_test_split(raw_X_train,
                                                                      raw_train_target,
                                                                      test_size=0.2)

        X_train = torch.from_numpy(X_train)
        train_target = torch.from_numpy(train_targets)
        X_val = torch.from_numpy(X_val)
        val_target = torch.from_numpy(val_targets)
        X_test = torch.from_numpy(X_test)
        test_target = torch.from_numpy(test_target.X.toarray())

        # TODO: Continue here (generate all the features)
        save_path = f"{self.data_path}/input_features/{self.fm}"
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        with gzip.open(f"{self.data_path}/input_features/{self.fm}/train_data_{self.spectral_parameter}.pkl.gz",
                       "wb") as f:
            pkl.dump((X_train, train_target), f)
        with gzip.open(f"{self.data_path}/input_features/{self.fm}/val_data_{self.spectral_parameter}.pkl.gz",
                       "wb") as f:
            pkl.dump((X_val, val_target), f)
        with gzip.open(f"{self.data_path}/input_features/{self.fm}/test_data_{self.spectral_parameter}.pkl.gz",
                       "wb") as f:
            pkl.dump((X_test, test_target), f)

        return X_train, train_target, X_val, val_target, X_test, test_target, ctrl_expr, self.all_perts_test

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

        train_input_expr = basal_ctrl_adata[random_train_mask, :].X.toarray()
        test_input_expr = basal_ctrl_adata[random_test_mask, :].X.toarray()

        print("\n\nInput expression data generated.\n\n")

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

        # save data as pickle without gzip
        with open(f"{self.data_path}/input_features/{self.fm}/train_data_{self.spectral_parameter}.pkl", "wb") as f:
            pkl.dump((X_train, train_target), f)
        with open(f"{self.data_path}/input_features/{self.fm}/val_data_{self.spectral_parameter}.pkl", "wb") as f:
            pkl.dump((X_val, val_target), f)
        with open(f"{self.data_path}/input_features/{self.fm}/test_data_{self.spectral_parameter}.pkl", "wb") as f:
            pkl.dump((X_test, test_target), f)

        raise HydraException(f"Completed preprocessing and featurisation of split {self.spectral_parameter}. Moving "
                             f"on the next multirun...")

        # return X_train, train_target, X_val, val_target, X_test, test_target

    @staticmethod
    def compute_correlations(pert, basal_ctrl_adata, all_gene_expression):
        pert_idx = basal_ctrl_adata.var_names.get_loc(pert)
        basal_expr_pert = basal_ctrl_adata.X[:, pert_idx].flatten()
        correlations = np.array(
            [pearsonr(basal_expr_pert, all_gene_expression[:, i])[0] for i in range(all_gene_expression.shape[1])])
        correlations[np.isnan(correlations)] = 0
        return pert, correlations

    def sg_pert_mask(self, mask, pert, idx, ctrl_adata):
        pert_idx = self.genes.index(pert)
        non_zero_indices = ctrl_adata[:, pert_idx].X.sum(axis=1).nonzero()[0]
        num_non_zeroes = len(non_zero_indices)

        if len(non_zero_indices) == 0:
            print(f"{pert} has no nonzero values in the control dataset! Kicking it from the analysis.")
            return mask
        elif len(non_zero_indices) < 500:
            sample_num = num_non_zeroes
        else:
            sample_num = 500

        sampled_indices = np.random.choice(non_zero_indices, sample_num, replace=False)
        mask[sampled_indices, idx] = True

        return mask

    def __getitem__(self, index):
        if self.stage == "train":
            return self.X_train[index], self.train_target[index], self.ctrl_expr[index]
        elif self.stage == "val":
            return self.X_val[index], self.val_target[index], self.ctrl_expr[index]
        elif self.stage == "test" and self.deg_dict is None:
            if self.all_perts_test is not None:
                return self.X_test[index], self.test_target[index], self.all_perts_test[index], self.ctrl_expr[index]
            else:
                return self.X_test[index], self.test_target[index], self.ctrl_expr[index]
        else:
            all_genes = self.basal_ctrl_adata.var.index.to_list()
            de_idx = [all_genes.index(gene) for gene in self.deg_dict[self.perturbation] if gene in all_genes]
            return self.X_test[index], self.test_target[index], {"de_idx": de_idx}, self.ctrl_expr[index]

    def __len__(self):
        if self.stage == "train":
            return len(self.X_train)
        elif self.stage == "val":
            return len(self.X_val)
        elif self.stage == "test":
            return len(self.X_test)
        else:
            raise ValueError(f"Invalid stage: {self.stage}. Must be 'train', 'val' or 'test'")
