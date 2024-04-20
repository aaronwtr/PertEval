import numpy as np

from typing import Any, Dict, Tuple

import torch
from lightning import LightningModule
from torch_geometric.data import Batch
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.regression import SpearmanCorrCoef, PearsonCorrCoef, MeanSquaredError

from .reproduction.gears.gears import GEARSNetwork
from gears.utils import loss_fct
from gears import PertData


class GEARSLitModule(LightningModule):
    """LightningModule wrapper for GEARS.

    A `LightningModule` implements 8 key methods:

    ```python
    def __init__(self):
    # Define initialization code here.

    def setup(self, stage):
    # Things to setup before each stage, 'fit', 'validate', 'test', 'predict'.
    # This hook is called on every process when using DDP.

    def training_step(self, batch, batch_idx):
    # The complete training step.

    def validation_step(self, batch, batch_idx):
    # The complete validation step.

    def test_step(self, batch, batch_idx):
    # The complete test step.

    def predict_step(self, batch, batch_idx):
    # The complete predict step.

    def configure_optimizers(self):
    # Define and configure optimizers and LR schedulers.
    ```

    Docs:
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """

    def __init__(
            self,

            net: GEARSNetwork,
            pertmodule: PertData,
            optimizer: Any,
            scheduler: Any,
            model_name: Any,
            compile: bool = False
    ) -> None:
        """Initialize a `GEARSLitModule`.

        :param net: The model to train.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.net = net

        # loss function
        self.criterion = loss_fct

        adata = pertmodule.pert_data.adata
        pert_full_id2pert = dict(adata.obs[['condition_name', 'condition']].values)
        self.dict_filter = {pert_full_id2pert[i]: j for i, j in adata.uns['non_zeros_gene_idx'].items()
                            if i in pert_full_id2pert}

        self.pert_list = pertmodule.pert_list

        self.ctrl_expression = torch.tensor(np.mean(adata.X[adata.obs.condition == 'ctrl'], axis=0)).reshape(-1, )

        self.test_results = {}
        self.pert_cat = []
        self.test_pred = []
        self.test_truth = []
        self.test_pred_de = []
        self.test_truth_de = []

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()

        # for tracking best so far validation pearson correlation
        self.val_loss_best = MaxMetric()

        self.test_spr = SpearmanCorrCoef()
        self.test_prs = PearsonCorrCoef()
        self.test_mse = MeanSquaredError()

        self.metric2fct = {
            'mse': self.test_mse,
            'pearson': self.test_prs,
            'spearman': self.test_spr,
        }

        self.net.model_initialize(pertmodule)

        self.net.to(self.device)

    def forward(self, x: torch.Tensor, pert_idx: list, batch: Batch) -> torch.Tensor:
        """Perform a forward pass through the model `self.net`.

        :param x: Flattened representation of GEARS input graphs.
        :param pert_idx: The index of the perturbation.
        :param batch: PyG Batch object.
        :return: A tensor of gene-level RNA expression.
        """
        return self.net(x, pert_idx, batch)

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()

    def model_step(
            self, batch: Batch
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """
        y = batch.y

        dir_lambda = self.net.direction_lambda
        preds = self.forward(batch.x, batch.pert_idx, batch)

        loss = self.criterion(preds, y, perts=batch.pert, ctrl=self.ctrl_expression.to(self.device),
                              dict_filter=self.dict_filter, direction_lambda=dir_lambda)
        return loss, preds, y

    def training_step(
            self, batch: Batch, batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        loss, preds, targets = self.model_step(batch)
        # update and log metrics
        self.train_loss(loss)
        self.log("train/loss", self.train_loss, on_step=True, on_epoch=True, prog_bar=True)

        # return loss or backpropagation will fail
        return loss

    def on_train_epoch_end(self) -> None:
        """Lightning hook that is called when a training epoch ends."""
        pass

    def validation_step(self, batch: Batch, batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, _, _ = self.model_step(batch)
        # update and log metrics
        self.val_loss(loss)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        """Lightning hook that is called when a validation epoch ends."""
        pass

    def test_step(self, batch: Batch, batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        self.pert_cat.extend(batch.pert)

        _, p, t = self.model_step(batch)
        self.test_pred.extend(p)
        self.test_truth.extend(t)

        # Differentially expressed genes
        for itr, de_idx in enumerate(batch.de_idx):
            self.test_pred_de.append(p[itr, de_idx])
            self.test_truth_de.append(t[itr, de_idx])

        # all genes
        self.test_results['pert_cat'] = np.array(self.pert_cat)
        pred = torch.stack(self.test_pred)
        truth = torch.stack(self.test_truth)
        self.test_results['pred'] = pred
        self.test_results['truth'] = truth

        pred_de = torch.stack(self.test_pred_de)
        truth_de = torch.stack(self.test_truth_de)
        self.test_results['pred_de'] = pred_de
        self.test_results['truth_de'] = truth_de

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        metrics = {}
        metrics_pert = {}

        for m in self.metric2fct.keys():
            metrics[m] = []
            metrics[m + '_de'] = []

        for pert in np.unique(self.test_results['pert_cat']):
            metrics_pert[pert] = {}
            p_idx = np.where(self.test_results['pert_cat'] == pert)[0]

            pert_preds = torch.tensor(self.test_results['pred'][p_idx].mean(0))
            pert_truth = torch.tensor(self.test_results['truth'][p_idx].mean(0))
            pert_truth_de = torch.tensor(self.test_results['truth_de'][p_idx].mean(0))
            pert_preds_de = torch.tensor(self.test_results['pred_de'][p_idx].mean(0))
            for m, fct in self.metric2fct.items():
                if m == 'pearson':
                    val = fct(pert_preds, pert_truth).item()
                    if np.isnan(val):
                        val = 0
                else:
                    val = fct(pert_preds, pert_truth).item()

                metrics_pert[pert][m] = val
                metrics[m].append(metrics_pert[pert][m])

            if pert != 'ctrl':
                for m, fct in self.metric2fct.items():
                    if m == 'pearson':
                        val = fct(pert_preds_de, pert_truth_de).item()
                        if np.isnan(val):
                            val = 0
                    else:
                        val = fct(pert_preds_de, pert_truth_de).item()

                    metrics_pert[pert][m + '_de'] = val
                    metrics[m + '_de'].append(metrics_pert[pert][m + '_de'])

            else:
                for m, fct in self.metric2fct.items():
                    metrics_pert[pert][m + '_de'] = 0

        for m in self.metric2fct.keys():
            stacked_metrics = np.stack(metrics[m])
            metrics[m] = np.mean(stacked_metrics)

            stacked_metrics_de = np.stack(metrics[m + '_de'])
            metrics[m + '_de'] = np.mean(stacked_metrics_de)

        metric_names = ['mse', 'pearson', 'spearman']

        for m in metric_names:
            self.log("test/" + m, metrics[m])
            self.log("test_de/" + m, metrics[m + '_de'])

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


if __name__ == "__main__":
    _ = GEARSLitModule(None, None)
