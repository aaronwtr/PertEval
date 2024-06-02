import os

from typing import Any, Dict, Optional
from pertpy import data as scpert_data

from lightning import LightningDataModule
from torch.utils.data import DataLoader

from src.data.perturb_dataset import PerturbData

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
            split: float = 0.00,
            replicate: int = 0,
            batch_size: int = 64,
            spectra_parameters: Optional[Dict[str, Any]] = None,
            num_workers: int = 0,
            pin_memory: bool = False,
            **kwargs: Any,
    ) -> None:
        """Initialize a `PertDataModule`.

        :param data_dir: The data directory. Defaults to `""`.
        :param data_name: The name of the dataset. Defaults to `"norman"`. Can pick from "norman", "gasperini", and
        "repogle".
        :param batch_size: The batch size. Defaults to `64`.
        :param num_workers: The number of workers. Defaults to `0`.
        :param pin_memory: Whether to pin memory. Defaults to `False`.
        """
        super().__init__()

        self.num_genes = None
        self.num_perts = None
        self.pert_data = None
        self.pertmodule = None
        self.adata = None
        self.spectra_parameters = spectra_parameters
        self.data_name = data_name
        self.fm = kwargs.get("fm", None)

        # check if split is float
        if isinstance(split, float):
            self.spectral_parameter = f"{split:.2f}_{str(replicate)}"
        elif isinstance(split, str):
            self.spectral_parameter = f"{split}_{str(replicate)}"
        else:
            raise ValueError("split must be a float or a string!")

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.data_path = os.path.join(data_dir, self.data_name)

        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)

        self.data_train: Optional[DataLoader] = None
        self.data_val: Optional[DataLoader] = None
        self.data_test: Optional[DataLoader] = None

        self.load_scpert_data = {
            "norman": "norman_2019_raw",
            "replogle_k562": "replogle_2022_k562_essential",
            "replogle_rpe1": "replogle_2022_rpe1",
        }

        self.batch_size_per_device = batch_size

        # need to call prepare and setup manually to guarantee proper model setup
        self.prepare_data()
        self.setup()

    def prepare_data(self) -> None:
        """Put all downloading and preprocessing logic that only needs to happen on one device here. Lightning ensures
        that `self.prepare_data()` is called only within a single process on CPU, so you can safely add your logic
        within. In case of multi-node training, the execution of this hook depends upon `self.prepare_data_per_node()`.

        Downloading:
        Currently, supports "norman", "replogle_k562, replogle_rpe1" datasets.

        Do not use it to assign state (self.x = y).
        """
        if self.data_name in ["norman", "replogle_k562", "replogle_rpe1"]:
            if f"{self.load_scpert_data[self.data_name]}.h5ad" not in os.listdir("data/"):
                scpert_loader = getattr(scpert_data, self.load_scpert_data[self.data_name])
                scpert_loader()
        else:
            raise ValueError(f"Data name {self.data_name} not recognized. Choose from: 'norman', "
                             f"'replogle_k562', or replogle_rpe1")

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
            scpert_loader = getattr(scpert_data, self.load_scpert_data[self.data_name])
            adata = scpert_loader()

            self.train_dataset = PerturbData(adata, self.data_path, self.spectral_parameter, self.spectra_parameters,
                                             self.fm, stage="train")
            self.val_dataset = PerturbData(adata, self.data_path, self.spectral_parameter, self.spectra_parameters,
                                           self.fm, stage="val")
            self.test_dataset = PerturbData(adata, self.data_path, self.spectral_parameter, self.spectra_parameters,
                                            self.fm, stage="test")

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

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


if __name__ == "__main__":
    _ = PertDataModule()
