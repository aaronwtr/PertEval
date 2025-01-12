import argparse
import torch
print(torch.__version__)
print(torch.version.cuda)

# Package imports
from gears import PertData, GEARS
from gears.utils import dataverse_download
from zipfile import ZipFile
import tarfile
import numpy as np
import pickle
from spectrae import SpectraDataset
import scanpy as sc
from sklearn.model_selection import train_test_split
import os
from tqdm import tqdm

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train GEARS model with custom splits.')
    parser.add_argument('--split_folder', type=str, required=True,
                        help='Path to the split folder containing train.pkl and test.pkl')
    parser.add_argument('--gears_path', type=str, default='/data/SBCS-BessantLab/martina/gears',
                        help='Path to the gears directory')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of epochs to train the model')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use for training (e.g., "cuda" or "cpu")')
    args = parser.parse_args()
    split_folder = args.split_folder
    gears_path = args.gears_path.rstrip('/')  # Remove trailing slash if any
    epochs = args.epochs
    device = args.device
    
    print(split_folder)

    # Ensure necessary directories exist
    if not os.path.exists(gears_path):
        os.makedirs(gears_path)

    # Download dataloader if not already present
    data_file = f'{gears_path}/norman_umi_go.tar.gz'
    if not os.path.exists(data_file):
        dataverse_download('https://dataverse.harvard.edu/api/access/datafile/6979957', data_file)
        with tarfile.open(data_file, 'r:gz') as tar:
            tar.extractall(path=gears_path)

    # Download model if not already present
    # model_file = os.path.join(gears_path, 'model.zip')
    # if not os.path.exists(model_file):
    #     dataverse_download('https://dataverse.harvard.edu/api/access/datafile/6979956', model_file)
    #     with ZipFile(model_file, 'r') as zip_ref:
    #         zip_ref.extractall(path=gears_path)

    # Load custom train test splits
    with open(f'{split_folder}/train.pkl', 'rb') as file:
        train_splits = pickle.load(file)

    with open(f'{split_folder}/test.pkl', 'rb') as file:
        test_splits = pickle.load(file)

    # Load adata
    adata = sc.read(f'{gears_path}/Norman_2019_raw.h5ad')

    # Filter genes
    nonzero_genes = (adata.X.sum(axis=0) > 5).A1
    filtered_adata = adata[:, nonzero_genes]
    single_gene_mask = [True if "," not in name else False for name in adata.obs['guide_ids']]
    sg_adata = filtered_adata[single_gene_mask, :]
    sg_adata.obs['condition'] = sg_adata.obs['guide_ids'].replace('', 'ctrl')

    genes = sg_adata.var['gene_symbols'].to_list()
    genes_and_ctrl = genes + ['ctrl']

    # Remove cells with perts not in the genes
    sg_pert_adata = sg_adata[sg_adata.obs['condition'].isin(genes_and_ctrl), :]

    # Create PerturbGraphData
    perturb_graph_data = PerturbGraphData(sg_pert_adata, 'norman')
    del nonzero_genes, filtered_adata, single_gene_mask, sg_adata, sg_pert_adata, genes, genes_and_ctrl

    # Function to get perturbation names
    def get_pert_names(pert_idxs):
        return [perturb_graph_data.samples[idx] for idx in pert_idxs]

    our_train_splits = get_pert_names(train_splits)
    our_test_splits = get_pert_names(test_splits)

    # Get splits in format needed for GEARS
    our_train_perts = [split + '+' + 'ctrl' for split in our_train_splits]
    our_test_perts = [split + '+' + 'ctrl' for split in our_test_splits]

    # Split our_train_perts into train and validation sets
    train_perts, val_perts = train_test_split(our_train_perts, test_size=0.2, random_state=42)

    # Load pert_data
    pert_data_folder = gears_path
    pert_data = PertData(pert_data_folder)
    data_name = 'norman_umi_go'
    pert_data.load(data_path = pert_data_folder + '/' + data_name)
    gear_perts = pert_data.adata.obs['condition'].cat.remove_unused_categories().cat.categories.tolist()

    # Filter perts
    def filter_perts(pert_list, gear_perts):
        filtered_perts = []
        for pert in pert_list:
            if pert in gear_perts:
                filtered_perts.append(pert)
            else:
                # Some perts might be 'ctrl+pert' instead of 'pert+ctrl'
                pn = pert.split('+')[0]
                new_pert_fmt = 'ctrl' + '+' + pn
                if new_pert_fmt in gear_perts:
                    filtered_perts.append(new_pert_fmt)
                else:
                    print(f"Perturbation {pert} not found in gear_perts.")
        return filtered_perts

    train_perts = filter_perts(train_perts, gear_perts)
    val_perts = filter_perts(val_perts, gear_perts)
    test_perts = filter_perts(our_test_perts, gear_perts)

    # Remove problematic perts
    problematic_perts = ["IER5L+ctrl", "SLC38A2+ctrl", "RHOXF2+ctrl"]
    for pert in problematic_perts:
        if pert in train_perts:
            train_perts.remove(pert)
        if pert in val_perts:
            val_perts.remove(pert)

    # Set up set2conditions
    set2conditions = {
        "train": train_perts,
        "val": val_perts,
        "test": test_perts
    }

    # Ensure that the sets are not empty
    if not train_perts:
        raise ValueError("Training set is empty after filtering.")
    if not val_perts:
        raise ValueError("Validation set is empty after filtering.")
    if not test_perts:
        raise ValueError("Test set is empty after filtering.")

    # Set up pert_data
    pert_data.set2conditions = set2conditions
    pert_data.split = "custom"
    pert_data.subgroup = None
    pert_data.seed = 1
    pert_data.get_dataloader(batch_size=32, test_batch_size=128)

    # Train the model
    gears_model = GEARS(pert_data, device=device)
    gears_model.model_initialize(hidden_size=64)
    gears_model.train(epochs=epochs)
