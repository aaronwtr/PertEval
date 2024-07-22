import os
import logging
import anndata
import gzip

import numpy as np
import pickle as pkl
import scanpy as sc

from tqdm import tqdm


class PertDataProcessor:
    def __init__(self, adata, data_path, data_name, unique_perts, fm='raw_expression', logger=None):
        self.adata = adata
        self.data_path = data_path
        self.data_name = data_name
        self.fm = fm
        self.gene_to_ensg = dict(zip(adata.var['gene_symbols'], adata.var_names))
        self.unique_perts = unique_perts
        self.fm_pert_data = None
        self.pert_adata = None
        self.ctrl_adata = None
        self.basal_ctrl_adata = None
        self.logger = logger or logging.getLogger(__name__)

    def process_data(self):
        try:
            if self.check_processed_files_exist() and self.fm != 'raw_expression':
                ctrl_expr, basal_ctrl_adata, pert_adata = self.load_embedded_data()
                return ctrl_expr, basal_ctrl_adata, pert_adata
            elif self.check_processed_files_exist() and self.fm == 'raw_expression':
                basal_ctrl_adata, pert_adata = self.load_and_process_final_data()
                return basal_ctrl_adata, pert_adata
            else:
                return self.process_and_save_data()

        except Exception as e:
            self.logger.error(f"Error processing data on line {e.__traceback__.tb_lineno}: {str(e)} ")

    def load_fm_embs(self):
        with gzip.open(f"{self.data_path}/embeddings/{self.data_name}_{self.fm}_fm_ctrl.pkl.gz",
                       "rb") as f:
            self.fm_ctrl_data = pkl.load(f)
        with gzip.open(f"{self.data_path}/embeddings/{self.data_name}_{self.fm}_fm_pert.pkl.gz", "rb") as f:
            self.fm_pert_data = pkl.load(f)

        fm_pert_data = {pert: emb for pert, emb in self.fm_pert_data.items() if emb.shape[0] > 0}
        self.unique_perts = list(fm_pert_data.keys())

        assert isinstance(self.fm_ctrl_data, (np.ndarray, anndata.AnnData)), ("fm_ctrl_data should be an array or an "
                                                                         "h5ad file!")

        if isinstance(self.fm_ctrl_data, anndata.AnnData):
            assert hasattr(self.fm_ctrl_data, 'obsm'), "fm_ctrl_data should have an attribute 'obsm'!"
            self.fm_ctrl_X = self.fm_ctrl_data.obsm['X']
        else:
            self.fm_ctrl_X = self.fm_ctrl_data

        assert isinstance(fm_pert_data, dict), ("fm_pert_data should be a dictionary with perturbed gene as key and"
                                                "embedding as value!")

    @staticmethod
    def filter_data(adata):
        pert_adata = adata[adata.obs['condition'] != 'ctrl', :]
        ctrl_adata = adata[adata.obs['condition'] == 'ctrl', :]
        return ctrl_adata, pert_adata

    def save_raw_data(self, ctrl_adata, pert_adata):
        self.pert_save_file(ctrl_adata, f"ctrl_{self.data_name}_raw_counts.h5ad")
        self.pert_save_file(pert_adata, f"pert_{self.data_name}_raw_counts.h5ad")

    def pert_save_file(self, data, filename):
        filepath = os.path.join(self.data_path, filename)
        if not os.path.exists(filepath):
            data.write(filepath, compression='gzip')
            self.logger.info(f"Saved {filename}")

    def check_processed_files_exist(self):
        files_to_check = [
            f"basal_ctrl_{self.data_name}_pp_filtered.h5ad",
            f"{self.data_name}_pp_pert_filtered.h5ad",
            f"{self.data_name}_pp_ctrl_filtered.h5ad",
            f"embed_basal_ctrl_{self.data_name}_pp_filtered"
        ]
        return all(os.path.exists(os.path.join(self.data_path, f)) for f in files_to_check)

    def process_and_save_data(self):
        self.logger.info("Processing and saving adata files...")

        sg_hvg_adata = self.preprocess_data(self.adata)
        self.ctrl_adata, self.pert_adata = self.filter_data(sg_hvg_adata)

        self.pert_save_file(self.pert_adata, f"{self.data_name}_pp_pert_filtered.h5ad")
        self.pert_save_file(self.ctrl_adata, f"{self.data_name}_pp_ctrl_filtered.h5ad")

        if self.fm != 'raw_expression':
            self.basal_ctrl_adata, ctrl_expr = self.generate_ctrl_embs()
            return ctrl_expr, self.basal_ctrl_adata, self.pert_adata
        else:
            self.basal_ctrl_adata = self.create_basal_control(self.ctrl_adata, self.pert_adata)
            self.pert_save_file(self.basal_ctrl_adata, f"basal_ctrl_{self.data_name}_pp_filtered.h5ad")
            self.save_raw_data(self.ctrl_adata, self.pert_adata)
            return self.basal_ctrl_adata, self.pert_adata

    def generate_ctrl_embs(self, subset_size=500):
        self.basal_ctrl_adata = self.create_basal_control(self.ctrl_adata, self.pert_adata)
        self.load_fm_embs()

        emb_perts = list(self.fm_pert_data.keys())
        pert_adata = self.pert_adata[self.pert_adata.obs['condition'].isin(emb_perts), :]

        ctrl_expr = self.basal_ctrl_adata[self.basal_ctrl_adata.obs['condition'].isin(emb_perts), :]
        ctrl_expr = ctrl_expr.X.toarray()
        with open(f"{self.data_path}/raw_expression_{self.data_name}_{self.fm}_pp_filtered.pkl", "wb") as f:
            pkl.dump(ctrl_expr, f)

        basal_ctrl_X = np.zeros((pert_adata.shape[0], self.fm_ctrl_X.shape[1]))
        for cell in tqdm(range(pert_adata.shape[0])):
            random_cells = np.random.choice(self.fm_ctrl_X.shape[0], subset_size)
            subset_X = self.fm_ctrl_X[random_cells, :]
            basal_ctrl_X[cell, :] = subset_X.mean(axis=0)

        basal_ctrl_X_empty = np.zeros((pert_adata.shape[0], self.fm_ctrl_X.shape[1]))
        basal_ctrl_adata = anndata.AnnData(X=basal_ctrl_X_empty, obs=pert_adata.obs)
        basal_ctrl_adata.obsm['X'] = basal_ctrl_X

        basal_ctrl_adata.write(f"{self.data_path}/embed_basal_ctrl_{self.data_name}_pp_filtered.h5ad")

        return basal_ctrl_adata, ctrl_expr

    def preprocess_data(self, adata):
        sc.pp.normalize_total(adata)
        sc.pp.log1p(adata)
        sc.pp.highly_variable_genes(adata, n_top_genes=2000)
        highly_variable_genes = adata.var_names[adata.var['highly_variable']]
        unique_perts_ensg = [self.gene_to_ensg[pert] for pert in self.unique_perts]
        missing_perts = list(set(unique_perts_ensg) - set(highly_variable_genes))
        combined_genes = list(set(highly_variable_genes) | set(missing_perts))
        return adata[:, combined_genes]

    @staticmethod
    def create_basal_control(ctrl_adata, pert_adata):
        ctrl_X = ctrl_adata.X.toarray()
        subset_size = 500
        basal_ctrl_X = np.zeros((pert_adata.shape[0], ctrl_X.shape[1]))

        for cell in tqdm(range(pert_adata.shape[0])):
            random_cells = np.random.choice(ctrl_X.shape[0], subset_size)
            subset_X = ctrl_X[random_cells, :]
            basal_ctrl_X[cell, :] = subset_X.mean(axis=0)

        return anndata.AnnData(X=basal_ctrl_X, obs=pert_adata.obs, var=ctrl_adata.var)

    def load_and_process_final_data(self):
        if self.fm == 'raw_expression':
            basal_ctrl_adata = sc.read_h5ad(f"{self.data_path}/basal_ctrl_{self.data_name}_pp_filtered.h5ad")
            pert_adata = sc.read_h5ad(f"{self.data_path}/{self.data_name}_pp_pert_filtered.h5ad")
            return basal_ctrl_adata, pert_adata
        else:
            ctrl_expr, basal_ctrl_adata, pert_adata = self.load_embedded_data()
            return ctrl_expr, basal_ctrl_adata, pert_adata

    def load_embedded_data(self):
        with open(f"{self.data_path}/raw_expression_{self.data_name}_{self.fm}_pp_filtered.pkl", "rb") as f:
            ctrl_expr = pkl.load(f)
        basal_ctrl_adata = sc.read_h5ad(f"{self.data_path}/embed_basal_ctrl_{self.data_name}_pp_filtered.h5ad")
        pert_adata = sc.read_h5ad(f"{self.data_path}/{self.data_name}_pp_pert_filtered.h5ad")
        return ctrl_expr, basal_ctrl_adata, pert_adata
