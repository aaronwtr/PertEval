import os
import torch

import numpy as np

from typing import Any, Dict, Optional
from pertpy import data as scpert_data

from lightning import LightningDataModule
from torch.utils.data import DataLoader

from src.utils.spectra.perturb import PerturbGraphData, SPECTRAPerturb

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(SCRIPT_DIR)
ROOT_DIR = os.path.dirname(SRC_DIR)
with open(f'{ROOT_DIR}/cache/data_dir_cache.txt', 'r') as f:
    DATA_DIR = f.read().strip()


class PertDataModule(LightningDataModule):
    """`LightningDataModule` for perturbation data. Based on GEARS PertData class, but adapted for PyTorch Lightning.

    A `LightningDataModule` implements 7 key methods:

    ```python
        def prepare_data(self):
        # Things to do on 1 GPU/TPU (not on every GPU/TPU in DDP).
        # Download data, pre-process, split, save to disk, etc...

        def setup(self, stage):
        # Things to do on every process in DDP.
        # Data loading, set variables, etc...

        def train_dataloader(self):
        # return train dataloader

        def val_dataloader(self):
        # return validation dataloader

        def test_dataloader(self):
        # return test dataloader

        def predict_dataloader(self):
        # return predict dataloader

        def teardown(self, stage):
        # Called on every process in DDP.
        # Clean up after fit or test.
    ```

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://lightning.ai/docs/pytorch/latest/data/datamodule.html
    """

    def __init__(
            self,
            data_dir: str = DATA_DIR,
            data_name: str = "norman",
            split: str = "0.00_0",
            batch_size: int = 64,
            spectra_parameters: Optional[Dict[str, Any]] = None,
            num_workers: int = 0,
            pin_memory: bool = False,
    ) -> None:
        """Initialize a `PertDataModule`.

        :param data_dir: The data directory. Defaults to `""`.
        :param data_name: The name of the dataset. Defaults to `"norman"`. Can pick from "norman", "gasperini", and
        "repogle".
        :param train_val_test_split: The train, validation and test split. Defaults to `(0.8, 0.05, 0.15)`.
        :param batch_size: The batch size. Defaults to `64`.
        :param num_workers: The number of workers. Defaults to `0`.
        :param pin_memory: Whether to pin memory. Defaults to `False`.
        """
        # TODO [ ]: Integrate scPerturb
        #           Procedure:
        #           [X] Select HVGs
        #            ------ DOES SPECTRA DO THIS? ------
        #           [X] Randomly pair non-perturbed control cells with perturbed cells (same type)
        #           [X] log2 transform the input and target values
        #           [X] subtract the control from the perturbed cells to get the perturbation effect
        #            ------ DOES SPECTRA DO THIS? ------
        #           [X] generate SPECTRA splits
        #           [ ] calculate foundation model embeddings for the input (control) cells
        #           [ ] train GEARS MLP decoder for predicting perturbation effect on the embeddings
        #           [ ] train MLP decoder for predicting perturbation effect on the embeddings
        #           [ ] train logistic regression model for predicting perturbation effect on the embeddings
        #           [ ] evaluate PCC -> AUSPC for perturbation effect magnitude
        #           [ ] evaluate MCC for perturbation effect direction (predicted up/down vs true up/down)
        # TODO [ ]: Train on one spectra train-test and process correctly
        # TODO [ ]: Setup multirun experiment to run on all spectra train-test splits
        super().__init__()

        self.num_genes = None
        self.num_perts = None
        self.pert_data = None
        self.pertmodule = None
        self.adata = None
        self.spectra_parameters = spectra_parameters
        self.data_name = data_name
        self.split = split

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.data_path = os.path.join(data_dir, self.data_name)

        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)

        self.data_train: Optional[DataLoader] = None
        self.data_val: Optional[DataLoader] = None
        self.data_test: Optional[DataLoader] = None

        self.batch_size_per_device = batch_size

        # need to call prepare and setup manually to guarantee proper model setup
        self.prepare_data()
        self.setup()

    def prepare_data(self) -> None:
        """Put all downloading and preprocessing logic that only needs to happen on one device here. Lightning ensures
        that `self.prepare_data()` is called only within a single process on CPU, so you can safely add your logic
        within. In case of multi-node training, the execution of this hook depends upon `self.prepare_data_per_node()`.

        Downloading:
        Currently, supports "gasperini", "norman", "repogle" datasets.

        Do not use it to assign state (self.x = y).
        """
        if self.data_name in ["norman", "gasperini", "repogle"]:
            if self.data_name == "norman":
                scpert_data.norman_2019()
            if self.data_name == "gasperini":
                scpert_data.gasperini_2019_atscale()
            if self.data_name == "repogle":
                scpert_data.replogle_2022_k562_gwps()
        else:
            raise ValueError(f"Data name {self.data_name} not recognized. Choose from: 'norman', 'gasperini', "
                             f"'repogle'")

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """
        # Divide batch size by the number of devices.
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices "
                    f"({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size

        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            adata = self.adata
            highly_variable_genes = adata.var_names[adata.var['highly_variable']]
            hv_pert_adata = adata[:, highly_variable_genes]
            single_gene_mask = [True if "+" not in name else False for name in hv_pert_adata.obs['perturbation_name']]

            sghv_pert_adata = hv_pert_adata[single_gene_mask, :]
            sghv_pert_adata.obs['condition'] = sghv_pert_adata.obs['perturbation_name'].replace('control', 'ctrl')
            ctrl_adata = sghv_pert_adata[sghv_pert_adata.obs['condition'] == 'ctrl', :]
            pert_adata = sghv_pert_adata[sghv_pert_adata.obs['condition'] != 'ctrl', :]

            perturb_graph_data = PerturbGraphData(sghv_pert_adata, self.data_name)

            sc_spectra = SPECTRAPerturb(perturb_graph_data, binary=False)
            sc_spectra.pre_calculate_spectra_properties(self.data_path)

            sparsification_step = self.spectra_parameters['sparsification_step']
            sparsification = ["{:.2f}".format(i) for i in np.arange(0, 1.05, float(sparsification_step))]
            self.spectra_parameters.pop('sparsification_step')
            self.spectra_parameters['number_repeats'] = int(self.spectra_parameters['number_repeats'])
            self.spectra_parameters['spectral_parameters'] = sparsification
            self.spectra_parameters['data_path'] = self.data_path + "/"

            if not os.path.exists(f"{self.data_path}/norman_SPECTRA_splits"):
                sc_spectra.generate_spectra_splits(**self.spectra_parameters)

            sp = self.split.split('_')[0]
            rpt = self.split.split('_')[1]
            train, test = sc_spectra.return_split_samples(sp, rpt, f"{self.data_path}/{self.data_name}")
            pert_list = perturb_graph_data.samples
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

            # todo
            #  [ ] process the control data (random sampling from control data for each perturbed data)
            #  [ ] get perturbed gene expression data targets
            #  [ ] concat control gene expression input and one hot encoded perturbation labels

            # gears implementation of sampling control: Xs = ctrl_adata[np.random.randint(0, len(ctrl_adata), num_samples), :].X.toarray()
            train_target_x = train_target.X.toarray()

            print('joe')

def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        return self.data_train

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        return self.data_val

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        return self.data_test

    def teardown(self, stage: Optional[str] = None) -> None:
        """Lightning hook for cleaning up after `trainer.fit()`, `trainer.validate()`,
        `trainer.test()`, and `trainer.predict()`.

        :param stage: The stage being torn down. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
            Defaults to ``None``.
        """
        pass

    def state_dict(self) -> Dict[Any, Any]:
        """Called when saving a checkpoint. Implement to generate and save the datamodule state.

        :return: A dictionary containing the datamodule state that you want to save.
        """
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Called when loading a checkpoint. Implement to reload datamodule state given datamodule
        `state_dict()`.

        :param state_dict: The datamodule state returned by `self.state_dict()`.
        """
        pass

    def get_pert_data(self):
        return self.pert_data


if __name__ == "__main__":
    _ = PertDataModule()
