import os

from typing import Any, Dict, Optional, Tuple, Union
from gears import PertData, GEARS
from ruamel.yaml import YAML

import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from src.utils.utils import zip_data_download_wrapper, find_root_dir


ROOT_DIR = find_root_dir(os.path.dirname(os.path.abspath(__file__)))
with open(f'{ROOT_DIR}/cache/data_dir_cache.txt', 'r') as f:
    DATA_DIR = f.read().strip()


class GEARSDataModule(LightningDataModule):
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
            batch_size: int = 64,
            num_workers: int = 0,
            pin_memory: bool = False,
    ) -> None:
        """Initialize a `PertDataModule`.

        :param data_dir: The data directory. Defaults to `""`.
        :param data_name: The name of the dataset. Defaults to `"norman"`. Can pick from "norman", "adamson", "dixit",
            "replogle_k562_essential" and "replogle_rpe1_essential".
        :param train_val_test_split: The train, validation and test split. Defaults to `(0.8, 0.05, 0.15)`.
        :param batch_size: The batch size. Defaults to `64`.
        :param num_workers: The number of workers. Defaults to `0`.
        :param pin_memory: Whether to pin memory. Defaults to `False`.
        """
        super().__init__()

        self.num_genes = None
        self.num_perts = None
        self.pert_data = None
        self.data_name = data_name

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.data_path = f"{data_dir}/{self.data_name}"
        # if not os.path.exists(self.data_path):
        #     os.makedirs(self.data_path)

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
        Currently, supports "adamson", "norman", "dixit", "replogle_k562_essential" and "replogle_rpe1_essential"
        datasets.

        Do not use it to assign state (self.x = y).
        """
        # TODO: Add support for downloading from a specified url
        print(f"Downloading {self.data_name} data...")
        if os.path.exists(self.data_path):
            print(f"Found local copy of {self.data_name} data...")
        elif self.data_name in ['norman', 'adamson', 'dixit', 'replogle_k562_essential', 'replogle_rpe1_essential']:
            ## load from harvard dataverse
            if self.data_name == 'norman':
                url = 'https://dataverse.harvard.edu/api/access/datafile/6154020'
            elif self.data_name == 'adamson':
                url = 'https://dataverse.harvard.edu/api/access/datafile/6154417'
            elif self.data_name == 'dixit':
                url = 'https://dataverse.harvard.edu/api/access/datafile/6154416'
            elif self.data_name == 'replogle_k562_essential':
                ## Note: This is not the complete dataset and has been filtered
                url = 'https://dataverse.harvard.edu/api/access/datafile/7458695'
            elif self.data_name == 'replogle_rpe1_essential':
                ## Note: This is not the complete dataset and has been filtered
                url = 'https://dataverse.harvard.edu/api/access/datafile/7458694'
            zip_data_download_wrapper(url, self.data_path)
            print(f"Successfully downloaded {self.data_name} data and saved to {self.data_path}")
        else:
            raise ValueError("data_name should be either 'norman', 'adamson', 'dixit', 'replogle_k562_essential' or "
                             "'replogle_rpe1_essential'")
        PertData(self.data_path)

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
            pert_data = PertData(self.data_path)
            pert_data.load(data_path=self.data_path)
            pert_data.prepare_split(split='simulation', seed=1)
            pert_data.get_dataloader(batch_size=self.batch_size_per_device, test_batch_size=128)
            self.pert_data = pert_data
            self.gene_list = pert_data.gene_names.values.tolist()
            self.pert_list = pert_data.pert_names.tolist()
            # calculating num_genes and num_perts for GEARS
            self.num_genes = len(self.gene_list)
            self.num_perts = len(self.pert_list)
            # adding num_genes and num_perts to hydra configs
            yaml = YAML()
            yaml.preserve_quotes = True
            yaml.width = 4096
            with open(f'{ROOT_DIR}/configs/model/gears.yaml', 'r') as f:
                yaml_data = yaml.load(f)
            yaml_data['net']['num_genes'] = self.num_genes
            yaml_data['net']['num_perts'] = self.num_perts
            with open(f'{ROOT_DIR}/configs/model/gears.yaml', 'w') as f:
                yaml.dump(yaml_data, f)
            dataloaders = pert_data.dataloader
            self.data_train = dataloaders['train_loader']
            self.data_val = dataloaders['val_loader']
            self.data_test = dataloaders['test_loader']

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
    _ = GEARSDataModule()
