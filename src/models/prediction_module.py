# src/models/components/simple_module.py

from typing import Any, Literal, Optional, Dict, Tuple
import torch
import torch.nn as nn
from lightning import LightningModule
from torchmetrics import MeanSquaredError, MeanMetric
from src.models.components.predictors import LinearRegressionModel, MLP

class PredictionModule(LightningModule):
    def __init__(
        self,
        net: torch.nn.Module,
        model_type: Literal["linear_regression", "mlp"] = "mlp",
        optimizer: torch.optim.Optimizer = torch.optim.Adam,
        criterion: Optional[torch.nn.Module] = nn.MSELoss(),
        compile: Optional[bool] = False,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    ) -> None:
        super().__init__()
        
        self.save_hyperparameters(logger=False) 
        
        self.net = net
        self.model_type = model_type ## saving placeholder in case different forward logic is required for different models
        self.criterion = criterion
        self.compile = compile

        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        self.train_mse = MeanSquaredError()
        self.val_mse = MeanSquaredError()
        self.test_mse = MeanSquaredError()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def model_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x, y = batch

        if x.dtype != torch.float32:
            x = x.to(torch.float32)

        if y.dtype != torch.float32:
            y = y.to(torch.float32)

        preds = self.forward(x)
        loss = self.criterion(preds, y)
        return loss, preds, y

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        loss, preds, targets = self.model_step(batch)
        self.train_loss(loss)
        self.train_mse(preds, targets)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/mse", self.train_mse, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        loss, preds, targets = self.model_step(batch)
        self.val_loss(loss)
        self.val_mse(preds, targets)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/mse", self.val_mse, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        loss, preds, targets = self.model_step(batch)
        self.test_loss(loss)
        self.test_mse(preds, targets)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
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
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}