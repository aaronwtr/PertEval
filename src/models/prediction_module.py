from typing import Any, Literal, Optional, Dict, Tuple, List
import torch
import torch.nn as nn
from lightning import LightningModule
from torchmetrics import MeanSquaredError, MeanMetric


class PredictionModule(LightningModule):
    def __init__(
            self,
            net: torch.nn.Module,
            model_type: Literal["mean", "linear_regression", "mlp"] = "mlp",
            optimizer: torch.optim.Optimizer = torch.optim.Adam,
            criterion: Optional[torch.nn.Module] = nn.MSELoss(),
            compile: Optional[bool] = False,
            scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    ) -> None:
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.net = net
        self.model_type = model_type    # saving placeholder in case different forward logic is required for different
                                        # models
        self.criterion = criterion
        self.compile = compile

        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        self.train_mse = MeanSquaredError()
        self.val_mse = MeanSquaredError()
        self.test_mse = MeanSquaredError()
        self.baseline_mse = MeanSquaredError()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    # noinspection PyTupleAssignmentBalance
    def model_step(self, batch: Tuple[torch.Tensor, torch.Tensor, Optional[dict], Optional[torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if len(batch) == 4:
            x, y, _, input_expr = batch
        elif len(batch) == 3:
            x, y, input_expr = batch
        else:
            x, y = batch
            input_expr = None

        if x.dtype != torch.float32:
            x = x.to(torch.float32)

        if y.dtype != torch.float32:
            y = y.to(torch.float32)

        if input_expr is None:
            input_expr = x[:, :x.shape[1] // 2]
        elif input_expr.dtype != torch.float32:
            input_expr = input_expr.to(torch.float32)

        preds = self.forward(x)
        pert_effect = y - input_expr
        loss = torch.sqrt(self.criterion(preds, pert_effect))

        return loss, preds, pert_effect

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        loss, preds, targets = self.model_step(batch)
        self.train_loss(loss)
        self.train_mse(preds, targets)
        self.log("train/mse", self.train_mse, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        loss, preds, targets = self.model_step(batch)
        self.val_loss(loss)
        self.val_mse(preds, targets)
        self.log("val/mse", self.val_mse, on_step=False, on_epoch=True, prog_bar=True)

    # noinspection PyTupleAssignmentBalance
    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor, Optional[list], Optional[torch.Tensor]],
                  batch_idx: int) -> None:
        de_dict = None
        if len(batch) == 4:
            x, y, de_dict, input_expr = batch
        elif len(batch) == 3:
            x, y, input_expr = batch
        else:
            x, y = batch
            input_expr = x[:, :x.shape[1] // 2]
        if not de_dict:
            loss, preds, targets = self.model_step((x, y, input_expr))
            self.test_mse(preds, targets)
            self.log("test/mse", self.test_mse, on_step=False, on_epoch=True, prog_bar=True)
        else:
            de_idx = de_dict['de_idx']
            loss, preds, targets = self.model_step((x, y, input_expr))
            num_genes = len(de_idx)
            mean_expr = torch.mean(input_expr, dim=0)
            mean_expr = mean_expr.repeat(targets.shape[0], 1)
            mean_eff = mean_expr - input_expr
            self.log("test/num_de_genes", num_genes, on_step=False, on_epoch=True, prog_bar=False)
            de_idx = torch.tensor([int(idx[0]) for idx in de_idx])
            de_idx = torch.tensor(de_idx)
            preds = preds[:, de_idx]
            targets = targets[:, de_idx]
            self.baseline_mse(mean_eff[:, de_idx], targets)
            self.test_mse(preds, targets)
            baseline_mse_value = self.baseline_mse.compute()
            test_mse_value = self.test_mse.compute()
            adjusted_mse = test_mse_value - baseline_mse_value
            self.log("test/adjusted_mse", adjusted_mse, on_step=False, on_epoch=True, prog_bar=True)

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
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/mse",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}
