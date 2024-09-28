from typing import Any, Literal, Optional, Dict, Tuple, List
import torch
import torch.nn as nn
from lightning import LightningModule
from torchmetrics import MeanSquaredError, MeanMetric
import pickle as pkl


class PredictionModule(LightningModule):
    def __init__(
            self,
            net: torch.nn.Module,
            model_type: Literal["mean", "linear_regression", "mlp"] = "mlp",
            optimizer: torch.optim.Optimizer = torch.optim.Adam,
            criterion: Optional[torch.nn.Module] = nn.MSELoss(),
            compile: Optional[bool] = False,
            scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
            mean_adjusted: Optional[bool] = False,
            data_name: Optional[str] = None,
            save_dir: Optional[str] = None
    ) -> None:
        super().__init__()

        self.save_dir = save_dir
        self.save_hyperparameters(logger=False)
        self.mean_adjusted = mean_adjusted
        self.data_name = data_name

        self.net = net
        self.model_type = model_type

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
            x, y, deg_dict, input_expr = batch
        else:
            x, y, input_expr = batch

        if x.dtype != torch.float32:
            x = x.to(torch.float32)

        if y.dtype != torch.float32:
            y = y.to(torch.float32)

        if input_expr.dtype != torch.float32:
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
            x, y, _de_dict_or_test_pert, input_expr = batch
            if isinstance(_de_dict_or_test_pert, dict):
                de_dict = _de_dict_or_test_pert
            else:
                test_perts = _de_dict_or_test_pert
                test_perts_idx = [i for i, pert in enumerate(test_perts) if '+' in pert]
                input_expr = input_expr[test_perts_idx, :]
                x = x[test_perts_idx, :]
                y = y[test_perts_idx, :]

        elif len(batch) == 3:
            x, y, _expr_or_de_dict = batch
            if isinstance(_expr_or_de_dict, dict):
                de_dict = _expr_or_de_dict
                input_expr = x[:, :x.shape[1] // 2]
            else:
                input_expr = _expr_or_de_dict
        else:
            x, y = batch
            input_expr = x
        if not de_dict:
            loss, preds, targets = self.model_step((x, y, input_expr))
            self.test_mse(preds, targets)
            self.log("test/mse", self.test_mse, on_step=False, on_epoch=True, prog_bar=True)
        else:
            de_idx = de_dict['de_idx']
            loss, preds, targets = self.model_step((x, y, de_idx, input_expr))
            de_idx = torch.tensor([int(idx[0]) for idx in de_idx])
            de_idx = torch.tensor(de_idx)
            preds = preds[:, de_idx]
            targets = targets[:, de_idx]
            self.test_mse(preds, targets)
            self.log("test/mse", self.test_mse, on_step=False, on_epoch=True, prog_bar=True)

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
