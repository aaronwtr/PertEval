#!/usr/bin/env python

import torch
import argparse
import os
import pickle
import numpy as np
from gears import PertData, GEARS
from gears.utils import dataverse_download
from zipfile import ZipFile
import tarfile
from spectrae import SpectraDataset
import scanpy as sc
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Define the PerturbGraphData class
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

def main():
    parser = argparse.ArgumentParser(description='Train GEARS model with custom splits.')
    parser.add_argument('--gears_path', type=str, default='/data/SBCS-BessantLab/martina/gears',
                        help='Path to GEARS data directory')
    parser.add_argument('--spectra_splits_dir', type=str, required=True,
                        help='Directory containing the spectra splits')
    parser.add_argument('--split_name', type=str, required=True,
                        help='Name of the split to use (e.g., SP_0.00_0)')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--test_batch_size', type=int, default=128, help='Batch size for testing')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use for training')
    parser.add_argument('--output_dir', type=str, default='./output', help='Directory to save outputs')
    args = parser.parse_args()

    # Print the arguments
    print("Arguments:")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    gears_path = args.gears_path.rstrip('/')  # Remove trailing slash if any

    # Download dataloader if not already present
    data_file = f'{gears_path}/norman_umi_go.tar.gz'
    if not os.path.exists(data_file):
        dataverse_download('https://dataverse.harvard.edu/api/access/datafile/6979957', data_file)
        with tarfile.open(data_file, 'r:gz') as tar:
            tar.extractall(path=gears_path)

    # Load the train and test splits
    spectra_splits_path = os.path.join(args.spectra_splits_dir, args.split_name)

    print(spectra_splits_path)


    with open(os.path.join(spectra_splits_path, 'train.pkl'), 'rb') as file:
        train_splits = pickle.load(file)

    with open(os.path.join(spectra_splits_path, 'test.pkl'), 'rb') as file:
        test_splits = pickle.load(file)

    # Load the dataset
    adata = sc.read(os.path.join(args.gears_path, 'Norman_2019_raw.h5ad'))

    # Filter genes with sufficient expression
    nonzero_genes = (adata.X.sum(axis=0) > 5).A1
    filtered_adata = adata[:, nonzero_genes]

    # Initialize lists to track conditions
    conditions = []

    # Process perturbations while preserving order
    for guide_id in filtered_adata.obs['guide_ids']:
        if "," in guide_id:  # Two-gene perturbation
            conditions.append(guide_id.replace(',', '+'))
        elif guide_id == "":  # Empty guide_id, treat as control
            conditions.append("ctrl")
        else:  # Single-gene perturbation
            conditions.append(guide_id)

    # Assign the processed conditions back to the AnnData object
    filtered_adata.obs['condition'] = conditions

    # Create a mask to keep only single and two-gene perturbations, excluding "ctrl"
    perturbation_mask = filtered_adata.obs['condition'] != 'ctrl'
    pert_adata = filtered_adata[perturbation_mask, :]

    pert_adata.obs['condition'] = pert_adata.obs['condition'].astype('category')

    # Generate the PerturbGraphData object
    perturb_graph_data = PerturbGraphData(pert_adata, 'norman')

    def get_pert_names(pert_idxs):
        return [perturb_graph_data.samples[idx] for idx in pert_idxs]

    # Split train_splits into train and validation indices
    # train_indices, val_indices = train_test_split(train_splits, test_size=0.2, random_state=42)

    # Get perturbation names for train, val, and test
    our_train_splits = get_pert_names(train_splits)
    #our_val_splits = get_pert_names(val_indices)
    our_test_splits = get_pert_names(test_splits)

    # Function to add '+ctrl' to single-gene perturbations
    def add_ctrl_to_single_gene_perts(pert_list):
        updated_list = [
            pert + '+ctrl' if '+' not in pert else pert  # Add "+ctrl" if no "+" exists in the perturbation
            for pert in pert_list
        ]
        return updated_list

    # Process the perturbations
    our_train_perts = add_ctrl_to_single_gene_perts(our_train_splits)
    test_perts = add_ctrl_to_single_gene_perts(our_test_splits)

    train_perts, val_perts = train_test_split(our_train_perts, test_size=0.2, random_state=42)

    # Load pert_data
    pert_data_folder = args.gears_path
    pert_data = PertData(pert_data_folder)
    data_name = 'norman_umi_go'
    pert_data.load(data_path=os.path.join(pert_data_folder, data_name))

    # Get the list of perturbations in GEARS data
    gear_perts = pert_data.adata.obs['condition'].cat.remove_unused_categories().cat.categories.tolist()

    # Function to adjust perturbations
    def adjust_perturbations(pert_list, gear_perts):
        adjusted_pert_list = []
        for pert in pert_list:
            if pert in gear_perts:
                adjusted_pert_list.append(pert)
                continue
            if '+' not in pert:  # Single-gene perturbation
                # Add "+ctrl" to single-gene perturbations
                pn = pert
                new_pert_fmt = 'ctrl' + '+' + pn
                if new_pert_fmt in gear_perts:
                    print(f"Reformatted single-gene perturbation: {pert} -> {new_pert_fmt}")
                    adjusted_pert_list.append(new_pert_fmt)
                else:
                    print(f"Perturbation {pert} not found in gear_perts after adding '+ctrl'. Skipping.")
            else:  # Two-gene perturbation
                # Switch order of genes for two-gene perturbations
                genes = pert.split('+')
                switched_pert = '+'.join(genes[::-1])  # Reverse the gene order
                if switched_pert in gear_perts:
                    print(f"Switched two-gene perturbation: {pert} -> {switched_pert}")
                    adjusted_pert_list.append(switched_pert)
                else:
                    print(f"Perturbation {pert} not found in gear_perts after switching order. Skipping.")
        return adjusted_pert_list

    # Adjust the perturbations
    train = adjust_perturbations(train_perts, gear_perts)
    val = adjust_perturbations(val_perts, gear_perts)
    test = adjust_perturbations(test_perts, gear_perts)

    # Create set2conditions
    set2conditions = {
        "train": train,
        "val": val,
        "test": test
    }

    # Ensure that the sets are not empty
    if not train:
        raise ValueError("Training set is empty after filtering.")
    if not val:
        raise ValueError("Validation set is empty after filtering.")
    if not test:
        raise ValueError("Test set is empty after filtering.")

    # Update pert_data
    pert_data.set2conditions = set2conditions
    pert_data.split = "custom"
    pert_data.subgroup = None
    pert_data.seed = 1

    # Get the dataloaders
    pert_data.get_dataloader(batch_size=args.batch_size, test_batch_size=args.test_batch_size)

    # Initialize and train the model
    gears_model = GEARS(pert_data, device=args.device)
    gears_model.model_initialize(hidden_size=64)
    gears_model.train(epochs=args.epochs)

    # Save the trained model
    model_save_path = os.path.join(args.output_dir, f'gears_model_{args.split_name}.pt')
    gears_model.save_model(model_save_path)
    print(f"Model saved to {model_save_path}")

if __name__ == '__main__':
    main()
